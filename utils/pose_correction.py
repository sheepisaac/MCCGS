import os

import torch
from torch import nn


def _skew(v):
    zero = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], zero], dim=-1),
        ],
        dim=-2,
    )


def _so3_exp(rotvec):
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp_min(1e-9)
    axis = rotvec / theta
    k = _skew(axis)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(rotvec.shape[:-1] + (3, 3))
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    return eye + sin_t * k + (1.0 - cos_t) * (k @ k)


class FramePoseCorrection(nn.Module):
    """Small per-frame SE(3) residuals used to refine unified camera poses."""

    def __init__(self, frame_ids, init_scale=1e-6):
        super().__init__()
        valid_ids = sorted({int(frame_id) for frame_id in frame_ids if int(frame_id) >= 0})
        if valid_ids:
            self.max_frame_id = max(valid_ids)
            self.register_buffer("valid_frame_ids", torch.tensor(valid_ids, dtype=torch.long))
            num_slots = self.max_frame_id + 1
        else:
            self.max_frame_id = -1
            self.register_buffer("valid_frame_ids", torch.empty(0, dtype=torch.long))
            num_slots = 1
        self.delta = nn.Parameter(torch.zeros(num_slots, 6))
        if init_scale > 0:
            with torch.no_grad():
                self.delta.normal_(mean=0.0, std=init_scale)

    @property
    def has_frames(self):
        return self.valid_frame_ids.numel() > 0

    def _delta_matrix(self, frame_id, dtype, device):
        if frame_id < 0 or frame_id >= self.delta.shape[0]:
            return None
        xi = self.delta[frame_id].to(device=device, dtype=dtype)
        rot = _so3_exp(xi[:3].unsqueeze(0))[0]
        trans = xi[3:]
        mat = torch.eye(4, dtype=dtype, device=device)
        mat[:3, :3] = rot
        mat[:3, 3] = trans
        return mat

    def apply_to_camera(self, camera):
        frame_id = int(getattr(camera, "frame_id", -1))
        base_world_view = getattr(camera, "base_world_view_transform", camera.world_view_transform)
        delta = self._delta_matrix(frame_id, base_world_view.dtype, self.delta.device)
        if delta is None:
            return camera

        base_w2c = base_world_view.to(self.delta.device).transpose(0, 1)
        corrected_w2c = delta @ base_w2c
        corrected_world_view = corrected_w2c.transpose(0, 1)
        projection = camera.projection_matrix.to(corrected_world_view.device)

        camera.world_view_transform = corrected_world_view
        camera.full_proj_transform = (
            corrected_world_view.unsqueeze(0).bmm(projection.unsqueeze(0))
        ).squeeze(0)
        camera.camera_center = corrected_world_view.inverse()[3, :3]
        return camera

    def regularization_loss(self, l2_weight=0.0, smooth_weight=0.0):
        loss = self.delta.new_tensor(0.0)
        if not self.has_frames:
            return loss
        valid = self.valid_frame_ids.to(self.delta.device)
        params = self.delta[valid]
        if l2_weight > 0:
            loss = loss + l2_weight * (params ** 2).mean()
        if smooth_weight > 0 and params.shape[0] > 1:
            loss = loss + smooth_weight * ((params[1:] - params[:-1]) ** 2).mean()
        return loss

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "delta": self.delta.detach().cpu(),
                "valid_frame_ids": self.valid_frame_ids.detach().cpu(),
                "max_frame_id": self.max_frame_id,
            },
            path,
        )

    def load(self, path):
        state = torch.load(path, map_location="cpu")
        delta = state["delta"]
        if delta.shape != self.delta.shape:
            raise RuntimeError(f"Pose correction shape mismatch: checkpoint {delta.shape}, module {self.delta.shape}")
        with torch.no_grad():
            self.delta.copy_(delta.to(self.delta.device))
