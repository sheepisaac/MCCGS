import os

import torch
from torch import nn
import torch.nn.functional as F


class MotionCompensationMLP(nn.Module):
    def __init__(self, in_dim=12, hidden_dim=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.residual_head = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        h = self.trunk(x)
        confidence_logit = self.confidence_head(h).squeeze(-1)
        residual = torch.tanh(self.residual_head(h))
        return confidence_logit, residual


class MotionCompensationController:
    """Codec-inspired Gaussian motion propagation for MCCGS.

    The controller does not aggressively create unrelated points. It learns a
    confidence-gated residual on top of the current 4DGS deformation motion and
    only propagates existing Gaussians along confident frame-to-frame motion.
    """

    def __init__(
        self,
        hidden_dim=64,
        lr=1e-4,
        sample_points=2048,
        neighbor_count=4,
        time_step=1.0 / 9.0,
        residual_scale=0.25,
        loss_weight_neighbor=1.0,
        loss_weight_residual=0.05,
        loss_weight_confidence=0.1,
        max_propagated_points=1024,
        confidence_threshold=0.65,
        propagation_scale=1.0,
        proposal_topk=0,
        proposal_quantile=1.0,
        proposal_min_score=-1.0,
        proposal_nms_alpha=0.0,
        birth_opacity_scale=0.35,
        birth_scale_shrink=0.9,
        hypothesis_initial_prob=0.999,
    ):
        self.model = MotionCompensationMLP(hidden_dim=hidden_dim).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.sample_points = sample_points
        self.neighbor_count = neighbor_count
        self.time_step = float(time_step)
        self.residual_scale = residual_scale
        self.loss_weight_neighbor = loss_weight_neighbor
        self.loss_weight_residual = loss_weight_residual
        self.loss_weight_confidence = loss_weight_confidence
        self.max_propagated_points = max_propagated_points
        self.confidence_threshold = confidence_threshold
        self.propagation_scale = propagation_scale
        self.proposal_topk = int(proposal_topk)
        self.proposal_quantile = float(proposal_quantile)
        self.proposal_min_score = float(proposal_min_score)
        self.proposal_nms_alpha = float(proposal_nms_alpha)
        self.birth_opacity_scale = float(birth_opacity_scale)
        self.birth_scale_shrink = float(birth_scale_shrink)
        self.hypothesis_initial_prob = float(hypothesis_initial_prob)
        self._last_cache = None
        self.probation_indices = None
        self.probation_evidence = 0.0
        self.probation_tests = 0
        self.probation_delta_history = []
        self.next_group_id = 1
        self.hypothesis_groups = {}

    def new_hypothesis_group(self, source="motion", birth_time=None):
        group_id = self.next_group_id
        self.next_group_id += 1
        self.hypothesis_groups[group_id] = {
            "source": source,
            "birth_time": None if birth_time is None else float(birth_time),
            "evidence": 0.0,
            "tests": 0,
            "mature": False,
        }
        return group_id

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.optimizer.step()

    def has_probation(self):
        return self.probation_indices is not None and self.probation_indices.numel() > 0

    def probation_mask(self, num_points, device):
        mask = torch.zeros(num_points, device=device, dtype=torch.bool)
        if not self.has_probation():
            return mask
        valid = self.probation_indices[self.probation_indices < num_points].long()
        if valid.numel() > 0:
            mask[valid] = True
        return mask

    def register_probation(self, start_index, count, group_id=None):
        if count <= 0:
            return
        device = self.probation_indices.device if self.probation_indices is not None else "cuda"
        new_indices = torch.arange(start_index, start_index + count, device=device, dtype=torch.long)
        if group_id is not None and int(group_id) not in self.hypothesis_groups:
            self.hypothesis_groups[int(group_id)] = {
                "source": "external",
                "birth_time": None,
                "evidence": 0.0,
                "tests": 0,
                "mature": False,
            }
        if self.probation_indices is None or self.probation_indices.numel() == 0:
            self.probation_indices = new_indices
            self.probation_evidence = 0.0
            self.probation_tests = 0
            self.probation_delta_history = []
        else:
            self.probation_indices = torch.cat([self.probation_indices, new_indices], dim=0)

    def active_hypothesis_group_ids(self, gaussians, max_groups=4):
        if not self.hypothesis_groups or gaussians._hypothesis_group_id.numel() != gaussians.get_xyz.shape[0]:
            return []
        probation = gaussians._hypothesis_state == 1
        group_ids = torch.unique(gaussians._hypothesis_group_id[probation]).detach().cpu().tolist()
        group_ids = [int(group_id) for group_id in group_ids if int(group_id) > 0]
        group_ids.sort(key=lambda group_id: self.hypothesis_groups.get(group_id, {}).get("tests", 0))
        return group_ids[:max(int(max_groups), 1)]

    def group_mask(self, gaussians, group_id):
        return (gaussians._hypothesis_state == 1) & (gaussians._hypothesis_group_id == int(group_id))

    def update_hypothesis_group(
        self,
        gaussians,
        group_id,
        delta_loss,
        hypothesis_lr=50.0,
        mature_prob=0.7,
        prune_prob=0.03,
    ):
        group_id = int(group_id)
        group = self.hypothesis_groups.setdefault(
            group_id,
            {"source": "unknown", "birth_time": None, "evidence": 0.0, "tests": 0, "mature": False},
        )
        group["evidence"] = 0.8 * float(group.get("evidence", 0.0)) + 0.2 * float(delta_loss)
        group["tests"] = int(group.get("tests", 0)) + 1

        mask = self.group_mask(gaussians, group_id)
        if mask.sum() == 0:
            self.hypothesis_groups.pop(group_id, None)
            return "empty", {}

        step = float(hypothesis_lr) * float(delta_loss)
        gaussians._existence_logit[mask] = (gaussians._existence_logit[mask] + step).clamp(-12.0, 8.0)
        prob = torch.sigmoid(gaussians._existence_logit[mask]).mean()
        mean_prob = float(prob.detach().cpu())
        decision = "testing"
        if mean_prob >= float(mature_prob):
            gaussians._hypothesis_state[mask] = 0
            group["mature"] = True
            decision = "matured"
        elif mean_prob <= float(prune_prob):
            decision = "rejected"

        stats = {
            "evidence": float(group["evidence"]),
            "tests": int(group["tests"]),
            "prob": mean_prob,
            "step": step,
            "count": int(mask.sum().item()),
            "source": group.get("source", "unknown"),
        }
        if decision in ("matured", "rejected"):
            self.hypothesis_groups.pop(group_id, None)
        return decision, stats

    def update_probation_evidence(
        self,
        delta_loss,
        min_tests,
        accept_threshold,
        reject_threshold,
        adaptive=False,
        adaptive_min_abs=2e-6,
        adaptive_mad_scale=1.5,
        sign_ratio=0.65,
        history_size=16,
    ):
        if not self.has_probation():
            return "none", {}
        self.probation_evidence = 0.8 * self.probation_evidence + 0.2 * float(delta_loss)
        self.probation_tests += 1
        self.probation_delta_history.append(float(delta_loss))
        if len(self.probation_delta_history) > history_size:
            self.probation_delta_history = self.probation_delta_history[-history_size:]

        threshold_pos = float(accept_threshold)
        threshold_neg = float(reject_threshold)
        positive_ratio = 0.0
        negative_ratio = 0.0
        noise_floor = 0.0
        if adaptive and self.probation_delta_history:
            history = torch.tensor(self.probation_delta_history, dtype=torch.float32)
            median = history.median()
            mad = (history - median).abs().median()
            noise_floor = max(float(adaptive_min_abs), float(mad) * float(adaptive_mad_scale))
            threshold_pos = noise_floor
            threshold_neg = -noise_floor
            positive_ratio = float((history > 0).float().mean().item())
            negative_ratio = float((history < 0).float().mean().item())

        stats = {
            "threshold_pos": threshold_pos,
            "threshold_neg": threshold_neg,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "noise_floor": noise_floor,
        }
        if self.probation_tests < min_tests:
            return "testing", stats
        if adaptive:
            if self.probation_evidence >= threshold_pos and positive_ratio >= sign_ratio:
                self.clear_probation()
                return "accepted", stats
            if self.probation_evidence <= threshold_neg and negative_ratio >= sign_ratio:
                return "rejected", stats
            return "testing", stats
        if self.probation_evidence >= accept_threshold:
            self.clear_probation()
            return "accepted", stats
        if self.probation_evidence <= reject_threshold:
            return "rejected", stats
        return "testing", stats

    def clear_probation(self):
        self.probation_indices = None
        self.probation_evidence = 0.0
        self.probation_tests = 0
        self.probation_delta_history = []

    def _sample_indices(self, gaussians):
        num_points = gaussians.get_xyz.shape[0]
        if num_points == 0:
            return None
        count = min(self.sample_points, num_points)
        if count == num_points:
            return torch.arange(num_points, device=gaussians.get_xyz.device)
        return torch.randperm(num_points, device=gaussians.get_xyz.device)[:count]

    def _deformed_xyz(self, gaussians, indices, time_value):
        xyz = gaussians.get_xyz[indices]
        scales = gaussians._scaling[indices]
        rotations = gaussians._rotation[indices]
        opacity = gaussians._opacity[indices]
        shs = gaussians.get_features[indices]
        times = torch.full((indices.shape[0], 1), float(time_value), device=xyz.device, dtype=xyz.dtype)
        means3d, _, _, _, _ = gaussians._deformation(xyz, scales, rotations, opacity, shs, times)
        return means3d

    def _make_features(self, gaussians, indices, base_motion, grad_signal, time_value):
        xyz = gaussians.get_xyz[indices]
        xyz_all = gaussians.get_xyz.detach()
        center = xyz_all.mean(dim=0, keepdim=True)
        scale = (xyz_all.max(dim=0, keepdim=True).values - xyz_all.min(dim=0, keepdim=True).values).clamp_min(1e-6)
        xyz_norm = (xyz - center) / scale

        opacity = gaussians.get_opacity[indices]
        scaling = gaussians.get_scaling[indices]
        motion_scale = base_motion.detach().norm(dim=-1, keepdim=True).median().clamp_min(1e-5)
        motion_norm = base_motion / motion_scale

        if grad_signal is None or grad_signal.shape[0] <= int(indices.max()):
            grad = torch.zeros((indices.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
        else:
            grad = grad_signal[indices].detach().norm(dim=-1, keepdim=True)
            grad = grad / grad.max().clamp_min(1e-6)

        time = torch.full((indices.shape[0], 1), float(time_value), device=xyz.device, dtype=xyz.dtype)
        return torch.cat([xyz_norm, time, opacity, scaling, motion_norm, grad], dim=-1), grad

    @torch.no_grad()
    def refresh_completion_cache(self, gaussians, grad_signal, time_value):
        """Build a Bayesian-style birth posterior from current training evidence.

        The score is a lightweight posterior proxy:
          confidence from the learned MLP
        + image-space gradient evidence as residual/unknown-region likelihood
        + local motion uncertainty from neighbor disagreement
        + low-opacity prior for under-explained regions.
        """
        indices = self._sample_indices(gaussians)
        if indices is None or indices.shape[0] < 2:
            self._last_cache = None
            return {}

        t0 = float(time_value)
        t1 = min(1.0, t0 + self.time_step)
        if t1 <= t0 + 1e-6:
            t0 = max(0.0, t0 - self.time_step)
            t1 = float(time_value)

        xyz_t0 = self._deformed_xyz(gaussians, indices, t0)
        xyz_t1 = self._deformed_xyz(gaussians, indices, t1)
        base_motion = xyz_t1 - xyz_t0
        features, grad = self._make_features(gaussians, indices, base_motion, grad_signal, t0)
        confidence_logit, residual = self.model(features)
        confidence = torch.sigmoid(confidence_logit)

        motion_mag = base_motion.norm(dim=-1, keepdim=True).median().clamp_min(1e-5)
        corrected_motion = base_motion + residual * motion_mag * self.residual_scale

        dist = torch.cdist(xyz_t0.detach(), xyz_t0.detach())
        k = min(self.neighbor_count + 1, dist.shape[0])
        _, nn_idx = torch.topk(dist, k=k, largest=False)
        nn_idx = nn_idx[:, 1:]
        del dist

        neighbor_motion = base_motion[nn_idx]
        disagreement = (base_motion[:, None, :] - neighbor_motion).norm(dim=-1).mean(dim=-1)
        uncertainty = disagreement / disagreement.quantile(0.9).clamp_min(1e-6)
        uncertainty = uncertainty.clamp(0.0, 1.0)

        residual_evidence = grad.squeeze(-1).clamp(0.0, 1.0)
        low_opacity = (1.0 - gaussians.get_opacity[indices].squeeze(-1)).clamp(0.0, 1.0)
        motion_evidence = base_motion.norm(dim=-1)
        motion_evidence = motion_evidence / motion_evidence.quantile(0.9).clamp_min(1e-6)
        motion_evidence = motion_evidence.clamp(0.0, 1.0)

        likelihood = (
            0.45 * residual_evidence
            + 0.25 * uncertainty
            + 0.20 * low_opacity
            + 0.10 * motion_evidence
        ).clamp(0.0, 1.0)
        posterior = (confidence * likelihood).clamp(0.0, 1.0)
        threshold = self._proposal_threshold(posterior)
        candidates = posterior >= threshold

        self._last_cache = {
            "indices": indices.detach(),
            "corrected_motion": corrected_motion.detach(),
            "confidence": confidence.detach(),
            "posterior": posterior.detach(),
            "grad": residual_evidence.detach(),
            "uncertainty": uncertainty.detach(),
            "candidate_count": int(candidates.sum().item()),
            "threshold": float(threshold),
        }
        return {
            "posterior_mean": float(posterior.mean().cpu()),
            "posterior_max": float(posterior.max().cpu()),
            "posterior_candidates": float(candidates.sum().cpu()),
            "posterior_threshold": float(threshold),
            "uncertainty_mean": float(uncertainty.mean().cpu()),
            "residual_evidence_mean": float(residual_evidence.mean().cpu()),
        }

    def compute_loss(self, gaussians, grad_signal, time_value):
        indices = self._sample_indices(gaussians)
        if indices is None or indices.shape[0] < 2:
            self._last_cache = None
            return None, {}

        t0 = float(time_value)
        t1 = min(1.0, t0 + self.time_step)
        if t1 <= t0 + 1e-6:
            t0 = max(0.0, t0 - self.time_step)
            t1 = float(time_value)

        with torch.no_grad():
            xyz_t0 = self._deformed_xyz(gaussians, indices, t0)
            xyz_t1 = self._deformed_xyz(gaussians, indices, t1)
            base_motion = xyz_t1 - xyz_t0
        features, grad = self._make_features(gaussians, indices, base_motion, grad_signal, t0)
        confidence_logit, residual = self.model(features)

        motion_mag = base_motion.detach().norm(dim=-1, keepdim=True).median().clamp_min(1e-5)
        corrected_motion = base_motion.detach() + residual * motion_mag * self.residual_scale
        confidence = torch.sigmoid(confidence_logit)

        with torch.no_grad():
            xyz_detached = xyz_t0.detach()
            dist = torch.cdist(xyz_detached, xyz_detached)
            k = min(self.neighbor_count + 1, dist.shape[0])
            _, nn_idx = torch.topk(dist, k=k, largest=False)
            nn_idx = nn_idx[:, 1:]
            del dist

        neighbor_motion = corrected_motion[nn_idx]
        motion_delta = corrected_motion[:, None, :] - neighbor_motion
        neighbor_conf = confidence[:, None] * confidence[nn_idx]
        neighbor_loss = (neighbor_conf * motion_delta.abs().mean(dim=-1)).mean()

        with torch.no_grad():
            base_neighbor = base_motion.detach()[nn_idx]
            disagreement = (base_motion.detach()[:, None, :] - base_neighbor).norm(dim=-1)
            disagreement = disagreement / disagreement.median().clamp_min(1e-6)
            target_conf = torch.exp(-disagreement).mean(dim=-1)
            target_conf = (0.5 * target_conf + 0.5 * grad.squeeze(-1)).clamp(0.0, 1.0)

        confidence_loss = F.binary_cross_entropy_with_logits(confidence_logit, target_conf)
        residual_loss = (residual ** 2).mean()
        loss = (
            self.loss_weight_neighbor * neighbor_loss
            + self.loss_weight_confidence * confidence_loss
            + self.loss_weight_residual * residual_loss
        )

        self._last_cache = {
            "indices": indices.detach(),
            "corrected_motion": corrected_motion.detach(),
            "confidence": confidence.detach(),
            "grad": grad.detach().squeeze(-1),
        }
        stats = {
            "neighbor_loss": float(neighbor_loss.detach().cpu()),
            "confidence_loss": float(confidence_loss.detach().cpu()),
            "residual_loss": float(residual_loss.detach().cpu()),
            "confidence_mean": float(confidence.detach().mean().cpu()),
            "confidence_max": float(confidence.detach().max().cpu()),
            "motion_mean": float(base_motion.detach().norm(dim=-1).mean().cpu()),
        }
        return loss, stats

    @torch.no_grad()
    def _proposal_threshold(self, score):
        if score is None or score.numel() == 0:
            return float(self.confidence_threshold)
        if self.proposal_min_score >= 0:
            min_score = self.proposal_min_score
        else:
            min_score = self.confidence_threshold
        if self.proposal_quantile < 1.0 and score.numel() > 1:
            quantile = min(max(self.proposal_quantile, 0.0), 1.0)
            quantile_score = float(torch.quantile(score.detach().float(), quantile).cpu())
            return max(float(min_score), quantile_score)
        return float(min_score)

    @torch.no_grad()
    def _select_proposals(self, score):
        threshold = self._proposal_threshold(score)
        keep = score >= threshold
        kept_count = int(keep.sum().item())
        stats = {
            "threshold": float(threshold),
            "gated_count": kept_count,
            "selected_count": 0,
            "target_topk": int(self.proposal_topk),
        }
        if kept_count == 0:
            return None, stats

        kept = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        kept_scores = score[kept]
        _, order = torch.sort(kept_scores, descending=True)
        limit = self.max_propagated_points
        if self.proposal_topk > 0:
            limit = min(limit, self.proposal_topk)
        selected = kept[order[:limit]]
        stats["selected_count"] = int(selected.numel())
        return selected, stats

    @torch.no_grad()
    def _apply_spatial_nms(self, selected, gaussians):
        if selected is None or selected.numel() == 0 or self.proposal_nms_alpha <= 0:
            return selected
        parent_indices = self._last_cache["indices"][selected].long()
        offsets = self._last_cache["corrected_motion"][selected] * self.propagation_scale
        candidate_xyz = gaussians.get_xyz[parent_indices].detach() + offsets
        candidate_scale = gaussians.get_scaling[parent_indices].detach().mean(dim=-1).clamp_min(1e-6)

        kept = []
        for local_idx in range(selected.numel()):
            if not kept:
                kept.append(local_idx)
                continue
            prev = torch.tensor(kept, device=selected.device, dtype=torch.long)
            distances = torch.norm(candidate_xyz[prev] - candidate_xyz[local_idx], dim=-1)
            radii = self.proposal_nms_alpha * torch.maximum(candidate_scale[prev], candidate_scale[local_idx])
            if not torch.any(distances < radii):
                kept.append(local_idx)
        keep_idx = torch.tensor(kept, device=selected.device, dtype=torch.long)
        return selected[keep_idx]

    @torch.no_grad()
    def last_candidate_stats(self):
        if self._last_cache is None:
            return {}
        score = self._last_cache.get("posterior", self._last_cache.get("confidence"))
        if score is None or score.numel() == 0:
            return {}
        selected, selection_stats = self._select_proposals(score)
        probation = int(self.probation_indices.numel()) if self.has_probation() else 0
        return {
            "score_mean": float(score.mean().detach().cpu()),
            "score_max": float(score.max().detach().cpu()),
            "candidate_count": int(selection_stats["gated_count"]),
            "selected_count": int(selection_stats["selected_count"]),
            "nms_count": int(selection_stats.get("nms_count", selection_stats["selected_count"])),
            "threshold": float(selection_stats["threshold"]),
            "target_topk": int(selection_stats["target_topk"]),
            "nms_alpha": float(self.proposal_nms_alpha),
            "probation_count": probation,
        }

    @torch.no_grad()
    def last_selection_stats(self):
        if self._last_cache is None:
            return {}
        return dict(self._last_cache.get("selection_stats", {}))

    @torch.no_grad()
    def propagate(self, gaussians, time_value=None):
        if self._last_cache is None or self.max_propagated_points <= 0:
            return 0
        score = self._last_cache.get("posterior", self._last_cache["confidence"])
        selected, selection_stats = self._select_proposals(score)
        if selected is None or selected.numel() == 0:
            self._last_cache["selection_stats"] = selection_stats
            return 0
        selected = self._apply_spatial_nms(selected, gaussians)
        selection_stats["nms_count"] = int(selected.numel())
        self._last_cache["selection_stats"] = selection_stats
        if selected.numel() == 0:
            return 0
        selected_indices = self._last_cache["indices"][selected]
        offsets = self._last_cache["corrected_motion"][selected] * self.propagation_scale
        start_index = gaussians.get_xyz.shape[0]
        group_id = self.new_hypothesis_group("motion", birth_time=time_value)
        born_points = gaussians.birth_from_existing(
            selected_indices,
            offsets,
            opacity_scale=self.birth_opacity_scale,
            scale_shrink=self.birth_scale_shrink,
            hypothesis_group_id=group_id,
            existence_prob=getattr(self, "hypothesis_initial_prob", 0.999),
        )
        self.register_probation(start_index, born_points, group_id=group_id)
        return born_points

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
