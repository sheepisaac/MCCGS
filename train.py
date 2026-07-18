#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy
from utils.pose_correction import FramePoseCorrection
from utils.gaussian_birth import GaussianBirthController
from utils.motion_compensation import MotionCompensationController

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def _camera_gt_image(viewpoint, dataset_type):
    if dataset_type == "PanopticSports":
        return viewpoint["image"].cuda()
    return viewpoint.original_image.cuda()


def _image_gradient_l1(rendered, gt):
    render_dx = rendered[:, :, 1:] - rendered[:, :, :-1]
    gt_dx = gt[:, :, 1:] - gt[:, :, :-1]
    render_dy = rendered[:, 1:, :] - rendered[:, :-1, :]
    gt_dy = gt[:, 1:, :] - gt[:, :-1, :]
    return (render_dx - gt_dx).abs().mean() + (render_dy - gt_dy).abs().mean()


def _image_gradient_map(image):
    gray = image[:3].mean(dim=0, keepdim=True)
    dx = torch.zeros_like(gray)
    dy = torch.zeros_like(gray)
    dx[:, :, 1:] = (gray[:, :, 1:] - gray[:, :, :-1]).abs()
    dy[:, 1:, :] = (gray[:, 1:, :] - gray[:, :-1, :]).abs()
    return (dx + dy).squeeze(0)


def _cvte_image_loss(rendered, gt, opt):
    l1_weight = float(getattr(opt, "mcc_verify_l1_weight", 1.0))
    edge_weight = float(getattr(opt, "mcc_verify_edge_weight", 0.0))
    loss = l1_weight * l1_loss(rendered, gt).mean()
    if edge_weight > 0:
        loss = loss + edge_weight * _image_gradient_l1(rendered, gt)
    return loss


def _pixel_l1_map(rendered, gt):
    return (rendered[:3] - gt[:3]).abs().mean(dim=0)


def _local_counterfactual_evidence(full, without, gt, opt, support_mask=None):
    full_l1 = _pixel_l1_map(full, gt)
    without_l1 = _pixel_l1_map(without, gt)
    contribution = (full[:3] - without[:3]).abs().mean(dim=0)
    residual = full_l1.detach()

    weight = contribution
    if float(getattr(opt, "mcc_hypothesis_residual_weight", 0.0)) > 0:
        residual_norm = residual / residual.quantile(0.95).clamp_min(1e-6)
        weight = weight * (1.0 + float(opt.mcc_hypothesis_residual_weight) * residual_norm.clamp(0.0, 4.0))
    if float(getattr(opt, "mcc_hypothesis_edge_weight", 0.0)) > 0:
        edge_residual = (_image_gradient_map(full) - _image_gradient_map(gt)).abs()
        edge_norm = edge_residual / edge_residual.quantile(0.95).clamp_min(1e-6)
        weight = weight * (1.0 + float(opt.mcc_hypothesis_edge_weight) * edge_norm.clamp(0.0, 4.0))

    flat_weight = weight.flatten()
    if flat_weight.numel() == 0 or float(flat_weight.max().detach().cpu()) <= 0.0:
        return torch.zeros((), device=full.device), {"support": 0.0, "raw_delta": 0.0, "weight_mean": 0.0}

    if support_mask is not None and support_mask.any():
        support = support_mask.to(device=weight.device, dtype=torch.bool)
    else:
        quantile = min(max(float(getattr(opt, "mcc_hypothesis_support_quantile", 0.90)), 0.0), 1.0)
        threshold = torch.quantile(flat_weight.float(), quantile)
        support = weight >= threshold
    min_support = int(getattr(opt, "mcc_hypothesis_support_min", 64))
    if int(support.sum().item()) < min_support:
        k = min(max(min_support, 1), flat_weight.numel())
        _, top_idx = torch.topk(flat_weight, k=k, largest=True)
        support = torch.zeros_like(flat_weight, dtype=torch.bool)
        support[top_idx] = True
        support = support.view_as(weight)

    support_weight = weight[support].clamp_min(1e-8)
    local_delta_map = without_l1[support] - full_l1[support]
    weighted_delta = (local_delta_map * support_weight).sum() / support_weight.sum().clamp_min(1e-8)
    raw_delta = local_delta_map.mean()
    if bool(getattr(opt, "mcc_hypothesis_normalize", True)):
        local_full = (full_l1[support] * support_weight).sum() / support_weight.sum().clamp_min(1e-8)
        weighted_delta = weighted_delta / local_full.detach().clamp_min(1e-4)
    stats = {
        "support": float(support.sum().detach().cpu()),
        "raw_delta": float(raw_delta.detach().cpu()),
        "weight_mean": float(support_weight.mean().detach().cpu()),
    }
    return weighted_delta, stats


@torch.no_grad()
def _deformed_group_xyz(gaussians, group_mask, time_value):
    indices = torch.nonzero(group_mask, as_tuple=False).squeeze(-1)
    if indices.numel() == 0:
        return None
    max_points = int(getattr(_deformed_group_xyz, "max_points", 128))
    if indices.numel() > max_points:
        indices = indices[torch.linspace(0, indices.numel() - 1, steps=max_points, device=indices.device).long()]
    xyz = gaussians.get_xyz[indices]
    scales = gaussians._scaling[indices]
    rotations = gaussians._rotation[indices]
    opacity = gaussians._opacity[indices]
    shs = gaussians.get_features[indices]
    times = torch.full((indices.shape[0], 1), float(time_value), device=xyz.device, dtype=xyz.dtype)
    means3d, _, _, _, _ = gaussians._deformation(xyz, scales, rotations, opacity, shs, times)
    return means3d


@torch.no_grad()
def _projection_support_mask(gaussians, group_mask, viewpoint, opt):
    height = int(viewpoint.image_height)
    width = int(viewpoint.image_width)
    if height <= 0 or width <= 0:
        return None

    _deformed_group_xyz.max_points = int(getattr(opt, "mcc_hypothesis_projection_max_points", 128))
    xyz = _deformed_group_xyz(gaussians, group_mask, getattr(viewpoint, "time", 0.0))
    if xyz is None or xyz.numel() == 0:
        return None

    ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=-1)
    proj = xyz_h @ viewpoint.full_proj_transform.to(device=xyz.device, dtype=xyz.dtype)
    w = proj[:, 3].clamp_min(1e-6)
    ndc = proj[:, :3] / w.unsqueeze(-1)
    valid = torch.isfinite(ndc).all(dim=-1) & (w > 1e-6)
    valid = valid & (ndc[:, 0] >= -1.2) & (ndc[:, 0] <= 1.2) & (ndc[:, 1] >= -1.2) & (ndc[:, 1] <= 1.2)
    if not valid.any():
        return None

    xs = ((ndc[valid, 0] + 1.0) * 0.5 * (width - 1)).round().long().clamp(0, width - 1)
    ys = ((1.0 - ndc[valid, 1]) * 0.5 * (height - 1)).round().long().clamp(0, height - 1)
    radius = max(int(getattr(opt, "mcc_hypothesis_projection_radius", 24)), 1)
    mask = torch.zeros((height, width), device=xyz.device, dtype=torch.bool)
    for x, y in zip(xs.tolist(), ys.tolist()):
        x0 = max(x - radius, 0)
        x1 = min(x + radius + 1, width)
        y0 = max(y - radius, 0)
        y1 = min(y + radius + 1, height)
        mask[y0:y1, x0:x1] = True

    max_area = float(getattr(opt, "mcc_hypothesis_projection_max_area", 0.25))
    area = float(mask.float().mean().detach().cpu())
    if area <= 0.0 or area > max_area:
        return None
    return mask


@torch.no_grad()
def _unproject_pixels_to_world(viewpoint, xs, ys, depths):
    width = float(viewpoint.image_width)
    height = float(viewpoint.image_height)
    x_ndc = ((xs.float() + 0.5) / width) * 2.0 - 1.0
    y_ndc = 1.0 - ((ys.float() + 0.5) / height) * 2.0
    dirs_cam = torch.stack(
        [
            x_ndc * np.tan(float(viewpoint.FoVx) * 0.5),
            y_ndc * np.tan(float(viewpoint.FoVy) * 0.5),
            torch.ones_like(x_ndc),
        ],
        dim=-1,
    )
    view_inv = viewpoint.world_view_transform.to(device=depths.device, dtype=depths.dtype).inverse()
    dirs_world = torch.nn.functional.normalize(dirs_cam.to(depths.dtype) @ view_inv[:3, :3], dim=-1)
    center = viewpoint.camera_center.to(device=depths.device, dtype=depths.dtype).unsqueeze(0)
    return center + dirs_world * depths.unsqueeze(-1)


@torch.no_grad()
def _residual_ray_birth(
    gaussians,
    motion_controller,
    viewpoint_cams,
    render_pkgs,
    rendered_images,
    gt_images,
    iteration,
    scene,
    opt,
    tb_writer,
):
    if not getattr(opt, "mcc_ray_birth", False):
        return 0
    if iteration < opt.mcc_ray_start or iteration > opt.mcc_ray_end:
        return 0
    if iteration % max(opt.mcc_ray_interval, 1) != 0:
        return 0
    if gaussians.get_xyz.shape[0] >= 360000:
        return 0

    max_points = min(int(opt.mcc_ray_max_points), 360000 - int(gaussians.get_xyz.shape[0]))
    if max_points <= 0:
        return 0

    ray_views = min(max(int(opt.mcc_ray_views), 1), len(viewpoint_cams))
    per_view = max(max_points // ray_views, 1)
    born_total = 0
    selected_total = 0
    score_max = 0.0

    for view_idx in range(ray_views):
        viewpoint = viewpoint_cams[view_idx]
        render_pkg = render_pkgs[view_idx]
        rendered = torch.clamp(rendered_images[view_idx].detach(), 0.0, 1.0)
        gt = gt_images[view_idx].detach()
        if rendered.dim() == 4:
            rendered = rendered.squeeze(0)
        if gt.dim() == 4:
            gt = gt.squeeze(0)
        gt = gt[:3]
        depth = render_pkg.get("depth", None)
        if depth is None:
            continue
        depth = depth.detach()
        if depth.dim() == 3:
            depth = depth.squeeze(0)

        residual = (rendered[:3] - gt).abs().mean(dim=0)
        if float(getattr(opt, "mcc_ray_edge_weight", 0.0)) > 0:
            edge_residual = (_image_gradient_map(rendered) - _image_gradient_map(gt)).abs()
            residual = residual + float(opt.mcc_ray_edge_weight) * edge_residual

        valid_depth = torch.isfinite(depth)
        valid_depth = valid_depth & (depth > float(opt.mcc_ray_depth_min)) & (depth < float(opt.mcc_ray_depth_max))
        if not valid_depth.any():
            continue

        scores = residual[valid_depth]
        if scores.numel() == 0:
            continue
        threshold = max(
            float(opt.mcc_ray_residual_threshold),
            float(torch.quantile(scores.float(), min(max(float(opt.mcc_ray_quantile), 0.0), 1.0)).cpu()),
        )
        mask = valid_depth & (residual >= threshold)
        candidate_count = int(mask.sum().item())
        if candidate_count == 0:
            continue

        candidate_scores = residual[mask]
        limit = min(per_view, candidate_count, max_points - born_total)
        if limit <= 0:
            break
        _, order = torch.topk(candidate_scores, k=limit, largest=True)
        coords = torch.nonzero(mask, as_tuple=False)[order]
        ys, xs = coords[:, 0], coords[:, 1]
        selected_depth = depth[ys, xs]
        if float(opt.mcc_ray_depth_jitter) > 0:
            selected_depth = selected_depth * (
                1.0 + torch.randn_like(selected_depth) * float(opt.mcc_ray_depth_jitter)
            )
        new_xyz = _unproject_pixels_to_world(viewpoint, xs, ys, selected_depth)

        visible = render_pkg["visibility_filter"].detach()
        parent_pool = torch.nonzero(visible, as_tuple=False).squeeze(-1)
        if parent_pool.numel() == 0:
            parent_pool = torch.arange(gaussians.get_xyz.shape[0], device=gaussians.get_xyz.device)
        if parent_pool.numel() > 8192:
            parent_pool = parent_pool[torch.randperm(parent_pool.numel(), device=parent_pool.device)[:8192]]
        parent_xyz = gaussians.get_xyz[parent_pool].detach()
        nearest = torch.cdist(new_xyz.detach(), parent_xyz).argmin(dim=1)
        parent_indices = parent_pool[nearest]

        start_index = gaussians.get_xyz.shape[0]
        group_id = motion_controller.new_hypothesis_group(
            "ray",
            birth_time=getattr(viewpoint, "time", 0.0),
        ) if getattr(opt, "mcc_hypothesis_gate", False) else 0
        born = gaussians.birth_from_external_points(
            new_xyz,
            parent_indices,
            opacity_scale=float(opt.mcc_ray_opacity_scale),
            scale_shrink=float(opt.mcc_ray_scale_shrink),
            hypothesis_group_id=group_id,
            existence_prob=float(getattr(opt, "mcc_hypothesis_initial_prob", 0.999))
            if getattr(opt, "mcc_hypothesis_gate", False)
            else 0.999,
        )
        if born > 0:
            motion_controller.register_probation(start_index, born, group_id=group_id if group_id > 0 else None)
            born_total += born
            selected_total += limit
            score_max = max(score_max, float(candidate_scores[order[:limit]].max().cpu()))

    if born_total > 0:
        print(
            f"\n[ITER {iteration}] MCC residual-ray birth added {born_total} points "
            f"(selected_pixels={selected_total}, max_score={score_max:.4f})"
        )
        if tb_writer:
            tb_writer.add_scalar("fine/mcc_ray_birth_points", born_total, iteration)
            tb_writer.add_scalar("fine/mcc_ray_birth_selected_pixels", selected_total, iteration)
            tb_writer.add_scalar("fine/mcc_ray_birth_score_max", score_max, iteration)
    return born_total


@torch.no_grad()
def _verify_hypothesis_groups(
    gaussians,
    motion_controller,
    cameras,
    iteration,
    render_func,
    pipe,
    background,
    stage,
    dataset_type,
    pose_correction,
    opt,
    tb_writer,
):
    group_ids = motion_controller.active_hypothesis_group_ids(
        gaussians,
        max_groups=getattr(opt, "mcc_hypothesis_max_groups_per_verify", 4),
    )
    if not group_ids:
        return False

    verify_views = min(max(int(getattr(opt, "mcc_verify_views", 1)), 1), len(cameras))
    start_idx = (iteration // max(opt.mcc_verify_interval, 1)) % len(cameras)
    stride = max(len(cameras) // verify_views, 1)
    selected_cameras = [cameras[(start_idx + view_idx * stride) % len(cameras)] for view_idx in range(verify_views)]

    full_losses = []
    full_renders = []
    full_gts = []
    for viewpoint in selected_cameras:
        if pose_correction is not None:
            pose_correction.apply_to_camera(viewpoint)
        gt = _camera_gt_image(viewpoint, dataset_type)[:3]
        full = torch.clamp(
            render_func(viewpoint, gaussians, pipe, background, stage=stage, cam_type=dataset_type)["render"],
            0.0,
            1.0,
        )
        full_losses.append(_cvte_image_loss(full, gt, opt))
        full_renders.append(full)
        full_gts.append(gt)
    full_loss = torch.stack(full_losses).mean()

    prune_mask = torch.zeros(gaussians.get_xyz.shape[0], device=gaussians.get_xyz.device, dtype=torch.bool)
    for group_id in group_ids:
        group_info = motion_controller.hypothesis_groups.get(int(group_id), {})
        birth_time = group_info.get("birth_time", None)
        group_mask = motion_controller.group_mask(gaussians, group_id)
        if group_mask.sum() == 0:
            continue
        saved_logit = gaussians._existence_logit[group_mask].clone()
        gaussians._existence_logit[group_mask] = -20.0
        without_losses = []
        local_evidence = []
        local_support = []
        local_raw = []
        backtrack_evidence = []
        for view_idx, viewpoint in enumerate(selected_cameras):
            gt = full_gts[view_idx]
            without = torch.clamp(
                render_func(viewpoint, gaussians, pipe, background, stage=stage, cam_type=dataset_type)["render"],
                0.0,
                1.0,
            )
            without_losses.append(_cvte_image_loss(without, gt, opt))
            if getattr(opt, "mcc_hypothesis_local_evidence", False):
                support_mask = None
                if getattr(opt, "mcc_hypothesis_support_mode", "counterfactual") == "projection":
                    support_mask = _projection_support_mask(gaussians, group_mask, viewpoint, opt)
                evidence, evidence_stats = _local_counterfactual_evidence(
                    full_renders[view_idx],
                    without,
                    gt,
                    opt,
                    support_mask=support_mask,
                )
                local_evidence.append(evidence)
                local_support.append(evidence_stats["support"])
                local_raw.append(evidence_stats["raw_delta"])
                if getattr(opt, "mcc_hypothesis_backtrack", False) and birth_time is not None:
                    view_time = float(getattr(viewpoint, "time", 0.0))
                    window = float(getattr(opt, "mcc_hypothesis_backtrack_window", 1.0))
                    if view_time <= float(birth_time) + 1e-6 and abs(view_time - float(birth_time)) <= window:
                        backtrack_evidence.append(evidence)
        gaussians._existence_logit[group_mask] = saved_logit

        without_loss = torch.stack(without_losses).mean()
        global_delta = float((without_loss - full_loss).detach().cpu())
        if local_evidence:
            local_delta_tensor = torch.stack(local_evidence).mean()
            delta_tensor = local_delta_tensor
            backtrack_delta = 0.0
            inconsistency = 0.0
            if getattr(opt, "mcc_hypothesis_backtrack", False) and backtrack_evidence:
                back_tensor = torch.stack(backtrack_evidence).mean()
                inconsistency_tensor = torch.clamp(-torch.stack(backtrack_evidence), min=0.0).mean()
                delta_tensor = (
                    delta_tensor
                    + float(getattr(opt, "mcc_hypothesis_backtrack_weight", 0.5)) * back_tensor
                    - float(getattr(opt, "mcc_hypothesis_backtrack_negative_weight", 0.25)) * inconsistency_tensor
                )
                backtrack_delta = float(back_tensor.detach().cpu())
                inconsistency = float(inconsistency_tensor.detach().cpu())
            delta_loss = float(delta_tensor.detach().cpu())
            support_mean = sum(local_support) / max(len(local_support), 1)
            raw_local_delta = sum(local_raw) / max(len(local_raw), 1)
        else:
            delta_loss = global_delta
            support_mean = 0.0
            raw_local_delta = global_delta
            backtrack_delta = 0.0
            inconsistency = 0.0
        decision, stats = motion_controller.update_hypothesis_group(
            gaussians,
            group_id,
            delta_loss,
            hypothesis_lr=getattr(opt, "mcc_hypothesis_lr", 50.0),
            mature_prob=getattr(opt, "mcc_hypothesis_mature_prob", 0.7),
            prune_prob=getattr(opt, "mcc_hypothesis_prune_prob", 0.03),
        )
        if decision == "rejected":
            prune_mask = prune_mask | group_mask
        if tb_writer:
            tb_writer.add_scalar(f"fine/mcc_hypothesis_group_{group_id}_delta", delta_loss, iteration)
            tb_writer.add_scalar(f"fine/mcc_hypothesis_group_{group_id}_global_delta", global_delta, iteration)
            tb_writer.add_scalar(f"fine/mcc_hypothesis_group_{group_id}_support", support_mean, iteration)
            tb_writer.add_scalar(f"fine/mcc_hypothesis_group_{group_id}_backtrack", backtrack_delta, iteration)
            tb_writer.add_scalar(f"fine/mcc_hypothesis_group_{group_id}_inconsistency", inconsistency, iteration)
            tb_writer.add_scalar(f"fine/mcc_hypothesis_group_{group_id}_prob", stats.get("prob", 0.0), iteration)
        print(
            f"\n[ITER {iteration}] CVTE group {decision}: "
            f"group={group_id}, source={stats.get('source', 'unknown')}, "
            f"count={stats.get('count', 0)}, delta={delta_loss:.6f}, "
            f"global={global_delta:.6f}, local_raw={raw_local_delta:.6f}, "
            f"back={backtrack_delta:.6f}, incons={inconsistency:.6f}, support={support_mean:.0f}, "
            f"prob={stats.get('prob', 0.0):.4f}, tests={stats.get('tests', 0)}"
        )

    rejected_points = int(prune_mask.sum().item())
    if rejected_points > 0:
        gaussians.prune_points(prune_mask)
        motion_controller._last_cache = None
        print(f"\n[ITER {iteration}] CVTE pruned {rejected_points} rejected hypothesis points")
    active = torch.nonzero(gaussians._hypothesis_state == 1, as_tuple=False).squeeze(-1)
    motion_controller.probation_indices = active if active.numel() > 0 else None
    return True


@torch.no_grad()
def _verify_probation_hypotheses(
    gaussians,
    motion_controller,
    cameras,
    iteration,
    render_func,
    pipe,
    background,
    stage,
    dataset_type,
    pose_correction,
    opt,
    tb_writer,
):
    if not opt.mcc_verify_hypotheses or not motion_controller.has_probation() or not cameras:
        return

    if getattr(opt, "mcc_hypothesis_gate", False):
        handled = _verify_hypothesis_groups(
            gaussians,
            motion_controller,
            cameras,
            iteration,
            render_func,
            pipe,
            background,
            stage,
            dataset_type,
            pose_correction,
            opt,
            tb_writer,
        )
        if handled:
            return

    probation_mask = motion_controller.probation_mask(gaussians.get_xyz.shape[0], gaussians.get_xyz.device)
    if probation_mask.sum() == 0:
        motion_controller.clear_probation()
        return

    verify_views = min(max(int(getattr(opt, "mcc_verify_views", 1)), 1), len(cameras))
    start_idx = (iteration // max(opt.mcc_verify_interval, 1)) % len(cameras)
    stride = max(len(cameras) // verify_views, 1)
    selected_cameras = [cameras[(start_idx + view_idx * stride) % len(cameras)] for view_idx in range(verify_views)]

    full_losses = []
    for viewpoint in selected_cameras:
        if pose_correction is not None:
            pose_correction.apply_to_camera(viewpoint)
        gt = _camera_gt_image(viewpoint, dataset_type)[:3]
        full = torch.clamp(
            render_func(viewpoint, gaussians, pipe, background, stage=stage, cam_type=dataset_type)["render"],
            0.0,
            1.0,
        )
        full_losses.append(_cvte_image_loss(full, gt, opt))

    saved_opacity = gaussians._opacity.data[probation_mask].clone()
    gaussians._opacity.data[probation_mask] = -20.0
    without_losses = []
    for viewpoint in selected_cameras:
        gt = _camera_gt_image(viewpoint, dataset_type)[:3]
        without = torch.clamp(
            render_func(viewpoint, gaussians, pipe, background, stage=stage, cam_type=dataset_type)["render"],
            0.0,
            1.0,
        )
        without_losses.append(_cvte_image_loss(without, gt, opt))
    gaussians._opacity.data[probation_mask] = saved_opacity

    full_loss = torch.stack(full_losses).mean()
    without_loss = torch.stack(without_losses).mean()
    delta_loss = float((without_loss - full_loss).detach().cpu())
    probation_count = int(probation_mask.sum().item())
    decision, decision_stats = motion_controller.update_probation_evidence(
        delta_loss,
        opt.mcc_verify_min_tests,
        opt.mcc_verify_accept_threshold,
        opt.mcc_verify_reject_threshold,
        adaptive=getattr(opt, "mcc_verify_adaptive", False),
        adaptive_min_abs=getattr(opt, "mcc_verify_adaptive_min_abs", 0.000002),
        adaptive_mad_scale=getattr(opt, "mcc_verify_adaptive_mad_scale", 1.5),
        sign_ratio=getattr(opt, "mcc_verify_sign_ratio", 0.65),
        history_size=getattr(opt, "mcc_verify_history", 16),
    )

    if tb_writer:
        tb_writer.add_scalar("fine/mcc_verify_delta_loss", delta_loss, iteration)
        tb_writer.add_scalar("fine/mcc_verify_evidence_ema", motion_controller.probation_evidence, iteration)
        tb_writer.add_scalar("fine/mcc_verify_tests", motion_controller.probation_tests, iteration)
        tb_writer.add_scalar("fine/mcc_verify_probation_points", probation_count, iteration)
        tb_writer.add_scalar("fine/mcc_verify_views", verify_views, iteration)
        for key, value in decision_stats.items():
            tb_writer.add_scalar(f"fine/mcc_verify_{key}", value, iteration)

    print(
        f"\n[ITER {iteration}] CVTE {decision}: "
        f"probation={probation_count}, delta={delta_loss:.6f}, "
        f"evidence_ema={motion_controller.probation_evidence:.6f}, "
        f"tests={motion_controller.probation_tests}, views={verify_views}, "
        f"thr=[{decision_stats.get('threshold_neg', opt.mcc_verify_reject_threshold):.6f}, "
        f"{decision_stats.get('threshold_pos', opt.mcc_verify_accept_threshold):.6f}], "
        f"pos={decision_stats.get('positive_ratio', 0.0):.2f}"
    )

    if decision == "accepted":
        print(f"\n[ITER {iteration}] CVTE accepted probation hypotheses (delta={delta_loss:.6f})")
    elif decision == "rejected":
        rejected_points = probation_count
        gaussians.prune_points(probation_mask)
        motion_controller.clear_probation()
        motion_controller._last_cache = None
        print(f"\n[ITER {iteration}] CVTE rejected and pruned {rejected_points} probation points (delta={delta_loss:.6f})")


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer, pose_correction=None, birth_controller=None, motion_controller=None):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    refine_pose_this_stage = (
        pose_correction is not None
        and opt.pose_refine
        and stage == "fine"
        and pose_correction.has_frames
    )
    pose_optimizer = None
    if refine_pose_this_stage:
        pose_optimizer = torch.optim.Adam(pose_correction.parameters(), lr=opt.pose_refine_lr)
        print(
            f"Pose refinement enabled: lr={opt.pose_refine_lr}, "
            f"active iters=[{opt.pose_refine_start}, {opt.pose_refine_end}], "
            f"frames={pose_correction.valid_frame_ids.numel()}"
        )
    birth_this_stage = birth_controller is not None and opt.gaussian_birth and stage == "fine"
    if birth_this_stage:
        print(
            f"Gaussian birth enabled: interval={opt.gaussian_birth_interval}, "
            f"active iters=[{opt.gaussian_birth_start}, {opt.gaussian_birth_end}], "
            f"max_new={opt.gaussian_birth_max_points}"
        )
    motion_this_stage = motion_controller is not None and opt.mcc_motion_comp and stage == "fine"
    if motion_this_stage:
        print(
            f"MCC motion compensation enabled: interval={opt.mcc_motion_interval}, "
            f"active iters=[{opt.mcc_motion_start}, {opt.mcc_motion_end}], "
            f"sample_points={opt.mcc_motion_sample_points}, "
            f"birth_opacity={opt.mcc_birth_opacity_scale}, "
            f"birth_scale={opt.mcc_birth_scale_shrink}"
        )


    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    # 
    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=16,collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    
    # dynerf, zerostamp_init
    # breakpoint()
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 
                            # 
    count = 0
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    count +=1
                    viewpoint_index = (count ) % len(video_cams)
                    if (count //(len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    # print(viewpoint_index)
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    # print(custom_cam.time, viewpoint_index, count)
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, cam_type=scene.dataset_type)["render"]

                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive) :
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        pose_refine_active = (
            pose_optimizer is not None
            and iteration >= opt.pose_refine_start
            and iteration <= opt.pose_refine_end
        )
        if pose_refine_active:
            pose_optimizer.zero_grad(set_to_none=True)
        motion_active = (
            motion_this_stage
            and iteration >= opt.mcc_motion_start
            and iteration <= opt.mcc_motion_end
        )
        if motion_active:
            motion_controller.zero_grad()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera

        # dynerf's branch
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size :    
                    
                viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
                if not viewpoint_stack :
                    viewpoint_stack =  temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue
        # print(len(viewpoint_cams))     
        # breakpoint()   
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        render_pkgs = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        for viewpoint_cam in viewpoint_cams:
            if refine_pose_this_stage:
                pose_correction.apply_to_camera(viewpoint_cam)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            render_pkgs.append(render_pkg)
            images.append(image.unsqueeze(0))
            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        # Loss
        # breakpoint()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm
        

        loss = Ll1
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        if pose_refine_active:
            loss += pose_correction.regularization_loss(opt.pose_refine_l2, opt.pose_refine_smooth)
        motion_stats = None
        if motion_active:
            mean_time = sum(float(getattr(cam, "time", 0.0)) for cam in viewpoint_cams) / max(len(viewpoint_cams), 1)
            visibility_signal = visibility_filter.float().unsqueeze(-1)
            motion_loss, motion_stats = motion_controller.compute_loss(gaussians, visibility_signal, mean_time)
            if motion_loss is not None:
                loss += opt.mcc_motion_loss_weight * motion_loss
                if tb_writer:
                    tb_writer.add_scalar("fine/mcc_motion_loss", float(motion_loss.detach().cpu()), iteration)
                    for key, value in motion_stats.items():
                        tb_writer.add_scalar(f"fine/mcc_motion_{key}", value, iteration)
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss
        
        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan, stop training without automatic reexec.")
            return
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        if motion_active:
            mean_time = sum(float(getattr(cam, "time", 0.0)) for cam in viewpoint_cams) / max(len(viewpoint_cams), 1)
            completion_stats = motion_controller.refresh_completion_cache(
                gaussians,
                viewspace_point_tensor_grad,
                mean_time,
            )
            if iteration % max(opt.mcc_motion_interval, 1) == 0:
                candidate_stats = motion_controller.last_candidate_stats()
                if candidate_stats:
                    print(
                        f"\n[ITER {iteration}] MCC proposal score: "
                        f"mean={candidate_stats['score_mean']:.4f}, "
                        f"max={candidate_stats['score_max']:.4f}, "
                        f"candidates={candidate_stats['candidate_count']} "
                        f"(thr={candidate_stats['threshold']:.4f}), "
                        f"selected={candidate_stats['selected_count']} "
                        f"(topk={candidate_stats['target_topk']}), "
                        f"nms_alpha={candidate_stats['nms_alpha']:.2f}, "
                        f"probation={candidate_stats['probation_count']}"
                    )
            if tb_writer:
                for key, value in completion_stats.items():
                    tb_writer.add_scalar(f"fine/mcc_completion_{key}", value, iteration)
        birth_features = None
        if birth_this_stage and iteration >= opt.gaussian_birth_start and iteration <= opt.gaussian_birth_end:
            mean_time = sum(float(getattr(cam, "time", 0.0)) for cam in viewpoint_cams) / max(len(viewpoint_cams), 1)
            birth_features, birth_loss = birth_controller.training_step(gaussians, viewspace_point_tensor_grad, mean_time)
            if tb_writer and birth_loss is not None:
                tb_writer.add_scalar("fine/gaussian_birth_mlp_loss", birth_loss, iteration)
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type, pose_correction if refine_pose_this_stage else None)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
                if refine_pose_this_stage:
                    pose_correction.save(os.path.join(scene.model_path, "pose_correction.pth"))
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration %  100 == 99) :
                    # breakpoint()
                        render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                    # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                if birth_this_stage \
                    and iteration >= opt.gaussian_birth_start \
                    and iteration <= opt.gaussian_birth_end \
                    and iteration % opt.gaussian_birth_interval == 0 \
                    and gaussians.get_xyz.shape[0] < 360000:
                    born_points = birth_controller.birth(gaussians, birth_features)
                    if born_points > 0:
                        print(f"\n[ITER {iteration}] Gaussian birth added {born_points} points")
                        if tb_writer:
                            tb_writer.add_scalar("fine/gaussian_birth_points", born_points, iteration)
                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()

            if motion_this_stage \
                and opt.mcc_verify_hypotheses \
                and iteration >= opt.mcc_motion_start \
                and iteration % opt.mcc_verify_interval == 0:
                heldout_cameras = test_cams if len(test_cams) > 0 else train_cams
                _verify_probation_hypotheses(
                    gaussians,
                    motion_controller,
                    heldout_cameras,
                    iteration,
                    render,
                    pipe,
                    background,
                    stage,
                    scene.dataset_type,
                    pose_correction if refine_pose_this_stage else None,
                    opt,
                    tb_writer,
                )

            if motion_active \
                and iteration % opt.mcc_motion_interval == 0 \
                and gaussians.get_xyz.shape[0] < 360000:
                candidate_stats = motion_controller.last_candidate_stats()
                mean_time = sum(float(getattr(cam, "time", 0.0)) for cam in viewpoint_cams) / max(len(viewpoint_cams), 1)
                propagated_points = motion_controller.propagate(gaussians, time_value=mean_time)
                if propagated_points > 0:
                    selection_stats = motion_controller.last_selection_stats()
                    if selection_stats:
                        print(
                            f"\n[ITER {iteration}] MCC motion propagation added {propagated_points} points "
                            f"(gated={selection_stats.get('gated_count', 0)}, "
                            f"selected={selection_stats.get('selected_count', 0)}, "
                            f"after_nms={selection_stats.get('nms_count', propagated_points)})"
                        )
                    else:
                        print(f"\n[ITER {iteration}] MCC motion propagation added {propagated_points} points")
                    if tb_writer:
                        tb_writer.add_scalar("fine/mcc_motion_propagated_points", propagated_points, iteration)
                elif candidate_stats:
                    print(
                        f"\n[ITER {iteration}] MCC motion propagation skipped: "
                        f"candidates={candidate_stats['candidate_count']} "
                        f"(thr={candidate_stats['threshold']:.4f}), "
                        f"selected={candidate_stats['selected_count']}"
                    )

            if motion_active \
                and getattr(opt, "mcc_ray_birth", False) \
                and gaussians.get_xyz.shape[0] < 360000:
                ray_birth_points = _residual_ray_birth(
                    gaussians,
                    motion_controller,
                    viewpoint_cams,
                    render_pkgs,
                    images,
                    gt_images,
                    iteration,
                    scene,
                    opt,
                    tb_writer,
                )
                if tb_writer and ray_birth_points > 0:
                    tb_writer.add_scalar("fine/mcc_ray_birth_total_points", ray_birth_points, iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if pose_refine_active:
                    pose_optimizer.step()
                if motion_active:
                    motion_controller.step()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")
                if birth_this_stage:
                    birth_controller.save(os.path.join(scene.model_path, "gaussian_birth_mlp.pth"))
                if motion_this_stage:
                    motion_controller.save(os.path.join(scene.model_path, "mcc_motion_comp_mlp.pth"))
def _collect_frame_ids(scene):
    frame_ids = []
    for camera_set in (scene.train_camera, scene.test_camera, scene.video_camera):
        source = getattr(camera_set, "dataset", [])
        for caminfo in source:
            frame_id = getattr(caminfo, "frame_id", -1)
            if frame_id is not None and int(frame_id) >= 0:
                frame_ids.append(int(frame_id))
    return frame_ids


def _collect_time_step(scene):
    times = []
    for cameras in (scene.getTrainCameras(), scene.getTestCameras(), scene.getVideoCameras()):
        for cam in cameras:
            times.append(float(getattr(cam, "time", 0.0)))
    unique = sorted(set(round(t, 8) for t in times))
    diffs = [b - a for a, b in zip(unique, unique[1:]) if b > a]
    return min(diffs) if diffs else 1.0


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    pose_correction = None
    if opt.pose_refine:
        pose_correction = FramePoseCorrection(_collect_frame_ids(scene)).cuda()
        print(f"Initialized frame pose correction with {pose_correction.valid_frame_ids.numel()} frame ids.")
    birth_controller = None
    if opt.gaussian_birth:
        birth_controller = GaussianBirthController(
            lr=opt.gaussian_birth_lr,
            max_new_points=opt.gaussian_birth_max_points,
            score_threshold=opt.gaussian_birth_score_threshold,
            offset_scale=opt.gaussian_birth_offset_scale,
            candidate_pool=opt.gaussian_birth_candidate_pool,
        )
    motion_controller = None
    if opt.mcc_motion_comp:
        motion_controller = MotionCompensationController(
            hidden_dim=opt.mcc_motion_hidden_dim,
            lr=opt.mcc_motion_lr,
            sample_points=opt.mcc_motion_sample_points,
            neighbor_count=opt.mcc_motion_neighbor_count,
            time_step=_collect_time_step(scene),
            residual_scale=opt.mcc_motion_residual_scale,
            loss_weight_neighbor=opt.mcc_motion_neighbor_weight,
            loss_weight_residual=opt.mcc_motion_residual_weight,
            loss_weight_confidence=opt.mcc_motion_confidence_weight,
            max_propagated_points=opt.mcc_motion_max_propagated_points,
            confidence_threshold=opt.mcc_motion_confidence_threshold,
            propagation_scale=opt.mcc_motion_propagation_scale,
            proposal_topk=opt.mcc_proposal_topk,
            proposal_quantile=opt.mcc_proposal_quantile,
            proposal_min_score=opt.mcc_proposal_min_score,
            proposal_nms_alpha=opt.mcc_proposal_nms_alpha,
            birth_opacity_scale=opt.mcc_birth_opacity_scale,
            birth_scale_shrink=opt.mcc_birth_scale_shrink,
            hypothesis_initial_prob=opt.mcc_hypothesis_initial_prob if opt.mcc_hypothesis_gate else 0.999,
        )
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer, pose_correction, birth_controller, motion_controller)
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer, pose_correction, birth_controller, motion_controller)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type, pose_correction=None):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if pose_correction is not None:
                        pose_correction.apply_to_camera(viewpoint)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000,14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
