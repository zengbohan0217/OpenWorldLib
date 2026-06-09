# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Diagonal attention mask generators for Sana-WM flexible attention.

These masks control which tokens can attend to each other in the
flex-attention / block-sparse attention mechanism.  The masks are
"shrinked" — computed at block granularity (a block is a group of
spatial tokens belonging to one frame) rather than per-token.
"""

import math

import torch


def _get_frame_id_and_token_id(seq_len: int, num_frames: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(frame_id, token_within_frame)`` for each block position."""
    tokens_per_frame = seq_len // num_frames
    frame_id = torch.arange(seq_len, dtype=torch.long) // tokens_per_frame
    token_id = torch.arange(seq_len, dtype=torch.long) % tokens_per_frame
    block_id = token_id // block_size
    return frame_id, block_id


def gen_log_mask_shrinked(
    N_pad: int,
    N: int,
    num_frames: int,
    block_size: int = 1,
) -> torch.Tensor:
    """Log-scale diagonal mask: nearby frames attend fully, far frames log-decay.

    The mask is ``(N_pad, N_pad)`` boolean (``True`` = allowed to attend).

    The log-scale means the attention window grows with frame distance at
    a logarithmic rate, so tokens in frame *t* can attend to tokens in
    frame *s* when ``|t - s| <= log_scale * log2(max(1, |t - s|))``.
    """
    mask = torch.zeros((N, N), dtype=torch.bool)
    tokens_per_frame = N // num_frames

    def _log_window(d: int) -> int:
        # Window grows ~logarithmically: at distance d, allow up to 2 * ceil(log2(d+1)) frames
        return max(tokens_per_frame, int(2 * tokens_per_frame * math.log2(d + 2)))

    for t in range(num_frames):
        t_start = t * tokens_per_frame
        t_end = (t + 1) * tokens_per_frame
        for s in range(num_frames):
            d = abs(t - s)
            window = _log_window(d)
            if d * tokens_per_frame <= window:
                s_start = s * tokens_per_frame
                s_end = (s + 1) * tokens_per_frame
                mask[t_start:t_end, s_start:s_end] = True

    padded = torch.zeros((N_pad, N_pad), dtype=torch.bool)
    padded[:N, :N] = mask
    return padded


def gen_linear_mask_shrinked(
    N_pad: int,
    N: int,
    num_frames: int,
    block_size: int = 1,
) -> torch.Tensor:
    """Linear diagonal mask: each token attends to tokens within a fixed
    linear window centred on its own frame.

    Window size is ``2 * tokens_per_frame`` frames centred on the query
    frame — equivalent to a sliding window of ``±1`` frame.
    """
    mask = torch.zeros((N, N), dtype=torch.bool)
    tokens_per_frame = N // num_frames
    window_half_frames = max(1, min(16, num_frames // 4))

    for t in range(num_frames):
        t_start = t * tokens_per_frame
        t_end = (t + 1) * tokens_per_frame
        s_low = max(0, t - window_half_frames)
        s_high = min(num_frames, t + window_half_frames + 1)
        for s in range(s_low, s_high):
            s_start = s * tokens_per_frame
            s_end = (s + 1) * tokens_per_frame
            mask[t_start:t_end, s_start:s_end] = True

    # Also allow all tokens to attend to frame 0 (first-frame conditioning)
    mask[:, :tokens_per_frame] = True

    padded = torch.zeros((N_pad, N_pad), dtype=torch.bool)
    padded[:N, :N] = mask
    return padded


def gen_truncated_mask_shrinked(
    N_pad: int,
    N: int,
    num_frames: int,
    block_size: int = 1,
    max_frame_distance: int = 8,
) -> torch.Tensor:
    """Truncated diagonal mask: each query frame can attend only to frames
    within ``max_frame_distance`` frames away (inclusive).

    This is the simplest variant — strictly local temporal attention.
    """
    mask = torch.zeros((N, N), dtype=torch.bool)
    tokens_per_frame = N // num_frames

    for t in range(num_frames):
        t_start = t * tokens_per_frame
        t_end = (t + 1) * tokens_per_frame
        s_low = max(0, t - max_frame_distance)
        s_high = min(num_frames, t + max_frame_distance + 1)
        for s in range(s_low, s_high):
            s_start = s * tokens_per_frame
            s_end = (s + 1) * tokens_per_frame
            mask[t_start:t_end, s_start:s_end] = True

    # Always allow full attention to frame 0 (conditioning frame)
    mask[:, :tokens_per_frame] = True

    padded = torch.zeros((N_pad, N_pad), dtype=torch.bool)
    padded[:N, :N] = mask
    return padded