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
        self._last_cache = None
        self.probation_indices = None
        self.probation_evidence = 0.0
        self.probation_tests = 0
        self.probation_delta_history = []

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

    def register_probation(self, start_index, count):
        if count <= 0:
            return
        device = self.probation_indices.device if self.probation_indices is not None else "cuda"
        new_indices = torch.arange(start_index, start_index + count, device=device, dtype=torch.long)
        if self.probation_indices is None or self.probation_indices.numel() == 0:
            self.probation_indices = new_indices
            self.probation_evidence = 0.0
            self.probation_tests = 0
            self.probation_delta_history = []
        else:
            self.probation_indices = torch.cat([self.probation_indices, new_indices], dim=0)

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
        candidates = posterior >= self.confidence_threshold

        self._last_cache = {
            "indices": indices.detach(),
            "corrected_motion": corrected_motion.detach(),
            "confidence": confidence.detach(),
            "posterior": posterior.detach(),
            "grad": residual_evidence.detach(),
            "uncertainty": uncertainty.detach(),
            "candidate_count": int(candidates.sum().item()),
        }
        return {
            "posterior_mean": float(posterior.mean().cpu()),
            "posterior_max": float(posterior.max().cpu()),
            "posterior_candidates": float(candidates.sum().cpu()),
            "posterior_threshold": float(self.confidence_threshold),
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
    def last_candidate_stats(self):
        if self._last_cache is None:
            return {}
        score = self._last_cache.get("posterior", self._last_cache.get("confidence"))
        if score is None or score.numel() == 0:
            return {}
        candidates = score >= self.confidence_threshold
        probation = int(self.probation_indices.numel()) if self.has_probation() else 0
        return {
            "score_mean": float(score.mean().detach().cpu()),
            "score_max": float(score.max().detach().cpu()),
            "candidate_count": int(candidates.sum().item()),
            "threshold": float(self.confidence_threshold),
            "probation_count": probation,
        }

    @torch.no_grad()
    def propagate(self, gaussians):
        if self._last_cache is None or self.max_propagated_points <= 0:
            return 0
        score = self._last_cache.get("posterior", self._last_cache["confidence"])
        keep = score >= self.confidence_threshold
        if keep.sum() == 0:
            return 0

        kept = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        kept_scores = score[kept]
        _, order = torch.sort(kept_scores, descending=True)
        selected = kept[order[: self.max_propagated_points]]
        selected_indices = self._last_cache["indices"][selected]
        offsets = self._last_cache["corrected_motion"][selected] * self.propagation_scale
        start_index = gaussians.get_xyz.shape[0]
        born_points = gaussians.birth_from_existing(
            selected_indices,
            offsets,
            opacity_scale=0.35,
            scale_shrink=0.9,
        )
        self.register_probation(start_index, born_points)
        return born_points

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
