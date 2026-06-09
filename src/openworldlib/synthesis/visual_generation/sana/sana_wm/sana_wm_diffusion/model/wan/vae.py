# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from dataclasses import dataclass
from typing import Optional

import torch
from omegaconf import OmegaConf

from ........base_models.diffusion_model.video.wan_2p1.modules.vae import WanVAE as _WanVAE

__all__ = [
    "WanVAE",
    "WanVAEConfig",
]


@dataclass
class WanVAEConfig:
    cache_dir: Optional[str] = None


class WanVAE(_WanVAE):
    """Sana-specific WanVAE wrapper with config and explicit device migration."""

    def __init__(self, z_dim=16, vae_pth="cache/vae_step_411000.pth", dtype=torch.float, device="cuda"):
        self.cfg: WanVAEConfig = OmegaConf.to_object(OmegaConf.structured(WanVAEConfig))
        super().__init__(z_dim=z_dim, vae_pth=vae_pth, dtype=dtype, device=device)

    def to(self, device):
        self.device = device
        self.model.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.scale = [self.mean, 1.0 / self.std]
