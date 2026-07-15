import os

import torch
from torch import nn
import torch.nn.functional as F


class GaussianBirthMLP(nn.Module):
    def __init__(self, in_dim=10, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.offset_head = nn.Linear(hidden_dim, 3)

    def forward(self, features):
        h = self.net(features)
        score = self.score_head(h).squeeze(-1)
        offset = torch.tanh(self.offset_head(h))
        return score, offset


class GaussianBirthController:
    """MLP-assisted Gaussian birth from high-residual/high-gradient candidates."""

    def __init__(
        self,
        hidden_dim=64,
        lr=1e-4,
        max_new_points=2048,
        score_threshold=0.6,
        offset_scale=0.5,
        candidate_pool=8192,
    ):
        self.model = GaussianBirthMLP(hidden_dim=hidden_dim).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.max_new_points = max_new_points
        self.score_threshold = score_threshold
        self.offset_scale = offset_scale
        self.candidate_pool = candidate_pool

    def _features(self, gaussians, grad_signal, time_value):
        xyz = gaussians.get_xyz.detach()
        if xyz.shape[0] == 0:
            return None
        xyz_center = xyz.mean(dim=0, keepdim=True)
        xyz_scale = (xyz.max(dim=0, keepdim=True).values - xyz.min(dim=0, keepdim=True).values).clamp_min(1e-6)
        xyz_norm = (xyz - xyz_center) / xyz_scale

        opacity = gaussians.get_opacity.detach()
        scaling = gaussians.get_scaling.detach()
        if grad_signal is None:
            grad = torch.zeros((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
        else:
            grad = grad_signal.detach().norm(dim=-1, keepdim=True)
            grad = grad / grad.max().clamp_min(1e-6)
        deform = getattr(gaussians, "_deformation_accum", torch.zeros_like(xyz)).detach().norm(dim=-1, keepdim=True)
        deform = deform / deform.max().clamp_min(1e-6)
        time = torch.full((xyz.shape[0], 1), float(time_value), device=xyz.device, dtype=xyz.dtype)
        return torch.cat([xyz_norm, time, opacity, scaling, grad, deform], dim=-1)

    def training_step(self, gaussians, grad_signal, time_value):
        features = self._features(gaussians, grad_signal, time_value)
        if features is None:
            return features, None
        with torch.no_grad():
            grad = features[:, -2]
            threshold = grad.mean() + grad.std()
            targets = (grad >= threshold).float()
            if targets.sum() < 1:
                targets[torch.argmax(grad)] = 1.0

        self.optimizer.zero_grad(set_to_none=True)
        scores, offsets = self.model(features.detach())
        loss = F.binary_cross_entropy_with_logits(scores, targets)
        loss = loss + 1e-3 * (offsets ** 2).mean()
        loss.backward()
        self.optimizer.step()
        return features, float(loss.detach().cpu())

    @torch.no_grad()
    def birth(self, gaussians, features):
        if features is None or features.shape[0] == 0 or self.max_new_points <= 0:
            return 0
        num_points = features.shape[0]
        if num_points > self.candidate_pool:
            grad = features[:, -2]
            _, pool_idx = torch.topk(grad, k=self.candidate_pool, largest=True)
            candidate_features = features[pool_idx]
            candidate_indices = pool_idx
        else:
            candidate_features = features
            candidate_indices = torch.arange(num_points, device=features.device)

        scores, offsets = self.model(candidate_features)
        probs = torch.sigmoid(scores)
        keep = probs >= self.score_threshold
        if keep.sum() == 0:
            _, top_idx = torch.topk(probs, k=min(self.max_new_points, probs.shape[0]), largest=True)
        else:
            kept_idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
            kept_scores = probs[kept_idx]
            _, order = torch.sort(kept_scores, descending=True)
            top_idx = kept_idx[order[: self.max_new_points]]

        selected_indices = candidate_indices[top_idx]
        selected_offsets = offsets[top_idx]
        selected_scale = gaussians.get_scaling.detach()[selected_indices].mean(dim=-1, keepdim=True)
        selected_offsets = selected_offsets * selected_scale * self.offset_scale
        return gaussians.birth_from_existing(selected_indices, selected_offsets)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
