# SPDX-License-Identifier: Apache-2.0
# Reuse FSDP utilities from shared WAN 2.2 base implementation.
from .......base_models.diffusion_model.video.wan_2p2.distributed.fsdp import (  # noqa: F401
    free_model,
    shard_model,
)
