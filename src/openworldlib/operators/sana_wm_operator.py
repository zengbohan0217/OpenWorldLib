import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from .base_operator import BaseOperator

# ---------------------------------------------------------------------------
# Constants (matching Sana-WM's fixed resolution)
# ---------------------------------------------------------------------------

TARGET_HEIGHT = 704
TARGET_WIDTH = 1280

ALLOWED_ACTION_KEYS: frozenset[str] = frozenset("wasdijkl")


# ============================================================================
# Sana action DSL → c2w (ported from inference_sana_wm.py)
# ============================================================================


def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def parse_action_string(action: str) -> list[list[str]]:
    """``"w-10,iw-5,none-3"`` → list of per-frame held-key lists."""
    cleaned = "".join(action.replace("，", ",").split())
    if not cleaned:
        raise ValueError("action string is empty")
    per_frame: list[list[str]] = []
    for segment in cleaned.split(","):
        if not segment or "-" not in segment:
            raise ValueError(
                f"Invalid action segment {segment!r}: expected '<keys>-<duration>'."
            )
        keys_part, dur_str = segment.rsplit("-", 1)
        if not dur_str.isdigit() or int(dur_str) <= 0:
            raise ValueError(
                f"Action segment {segment!r} has a non-positive duration {dur_str!r}."
            )
        n = int(dur_str)
        keys_lower = keys_part.lower()
        if keys_lower == "none":
            keys: list[str] = []
        else:
            bad = sorted({c for c in keys_lower if c not in ALLOWED_ACTION_KEYS})
            if bad:
                raise ValueError(
                    f"Action segment {segment!r} contains unknown keys {bad}; "
                    f"allowed: {''.join(sorted(ALLOWED_ACTION_KEYS))}."
                )
            keys = sorted(set(keys_lower))
        per_frame.extend([list(keys) for _ in range(n)])
    return per_frame


def action_string_to_c2w(
    action: str,
    *,
    translation_speed: float = 0.05,
    rotation_speed_deg: float = 1.2,
    pitch_limit_deg: float = 85.0,
) -> np.ndarray:
    """Roll out a ``(N+1, 4, 4)`` camera-to-world trajectory from an action string.

    DSL: ``<keys>-<frames>`` segments joined by commas.
    ``wasd`` = translate, ``ijkl`` = pitch/yaw.
    OpenCV convention (``+X right, +Y down, +Z forward``).
    """
    per_frame = parse_action_string(action)
    rotate_rad = math.radians(rotation_speed_deg)
    pitch_limit_rad = math.radians(pitch_limit_deg)
    current = np.eye(4, dtype=np.float64)
    poses = [current.copy()]
    current_pitch = 0.0

    for keys in per_frame:
        held = set(keys)
        R = current[:3, :3]
        T_ = current[:3, 3]

        pitch_delta = (rotate_rad if "i" in held else 0.0) - (rotate_rad if "k" in held else 0.0)
        new_pitch = current_pitch + pitch_delta
        if not (-pitch_limit_rad <= new_pitch <= pitch_limit_rad):
            pitch_delta = 0.0
        else:
            current_pitch = new_pitch

        yaw_delta = (rotate_rad if "l" in held else 0.0) - (rotate_rad if "j" in held else 0.0)
        R_new = _rot_y(yaw_delta) @ R @ _rot_x(pitch_delta)

        forward = R_new[:, 2].copy()
        forward[1] = 0.0
        right = R_new[:, 0].copy()
        right[1] = 0.0
        fn = float(np.linalg.norm(forward))
        rn = float(np.linalg.norm(right))
        if fn > 0:
            forward /= fn + 1e-6
        if rn > 0:
            right /= rn + 1e-6
        move = np.zeros(3, dtype=np.float64)
        if "w" in held:
            move += forward * translation_speed
        if "s" in held:
            move -= forward * translation_speed
        if "d" in held:
            move += right * translation_speed
        if "a" in held:
            move -= right * translation_speed

        current = np.eye(4, dtype=np.float64)
        current[:3, :3] = R_new
        current[:3, 3] = T_ + move
        poses.append(current.copy())

    return np.stack(poses, axis=0).astype(np.float32)


# ============================================================================
# Intrinsics helpers (from inference_sana_wm.py)
# ============================================================================


def load_intrinsics(path: Path | str, num_frames: int) -> np.ndarray:
    """Load ``(num_frames, 4)`` intrinsics from a ``.npy`` file.

    Accepts arrays shaped ``(3, 3)``, ``(F, 3, 3)``, or ``(4,)``.
    """
    path = Path(path)
    arr = np.load(str(path)).astype(np.float32)
    if arr.shape == (4,):
        return np.broadcast_to(arr, (num_frames, 4)).copy()
    if arr.shape == (3, 3):
        v = np.array([arr[0, 0], arr[1, 1], arr[0, 2], arr[1, 2]], dtype=np.float32)
        return np.broadcast_to(v, (num_frames, 4)).copy()
    if arr.ndim == 3 and arr.shape[1:] == (3, 3) and arr.shape[0] >= num_frames:
        K = arr[:num_frames]
        return np.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], axis=1)
    raise ValueError(
        f"Unsupported intrinsics shape {arr.shape} for num_frames={num_frames}; "
        "expected (3,3), (F,3,3), or (4,)."
    )


def estimate_intrinsics_with_pi3x(image: Image.Image, device: torch.device) -> np.ndarray:
    """Estimate ``(fx, fy, cx, cy)`` from a single image using Pi3X.

    Requires ``pi3`` to be installed.  The image is temporarily resized to
    a Pi3X-friendly resolution; returned intrinsics are scaled back to the
    original image size.
    """
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import recover_intrinsic_from_rays_d

    import gc

    W_orig, H_orig = image.size
    pixel_limit = 255_000
    scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1.0
    W_t, H_t = W_orig * scale, H_orig * scale
    k, m = max(1, round(W_t / 14)), max(1, round(H_t / 14))
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_t / H_t:
            k -= 1
        else:
            m -= 1
    W_model, H_model = max(1, k) * 14, max(1, m) * 14
    resized = image.resize((W_model, H_model), Image.Resampling.LANCZOS)
    tensor = T.ToTensor()(resized).unsqueeze(0).unsqueeze(0).to(device)

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
    model.disable_multimodal()
    model.requires_grad_(False)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        out = model(imgs=tensor)
    rays_d = torch.nn.functional.normalize(out["local_points"], dim=-1)
    K_model = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)
    K_model_np = K_model[0, 0].detach().cpu().float().numpy()

    sx, sy = W_orig / W_model, H_orig / H_model
    fx, fy = float(K_model_np[0, 0] * sx), float(K_model_np[1, 1] * sy)
    cx, cy = float(K_model_np[0, 2] * sx), float(K_model_np[1, 2] * sy)

    # Free Pi3X memory
    del model, out, K_model, rays_d, tensor
    torch.cuda.empty_cache()
    gc.collect()

    return np.array([fx, fy, cx, cy], dtype=np.float32)


def transform_intrinsics_for_crop(
    intrinsics_vec4: np.ndarray,
    src_size: tuple[int, int],
    resized_size: tuple[int, int],
    crop_offset: tuple[int, int],
) -> np.ndarray:
    """Adjust ``[fx, fy, cx, cy]`` to match a resize-then-center-crop image."""
    src_w, src_h = src_size
    rw, rh = resized_size
    cl, ct = crop_offset
    sx, sy = rw / src_w, rh / src_h
    out = intrinsics_vec4.copy()
    out[..., 0] *= sx
    out[..., 2] = out[..., 2] * sx - cl
    out[..., 1] *= sy
    out[..., 3] = out[..., 3] * sy - ct
    return out


# ============================================================================
# Image preprocessing
# ============================================================================


def resize_and_center_crop(
    image: Image.Image, target_h: int = TARGET_HEIGHT, target_w: int = TARGET_WIDTH
) -> tuple[Image.Image, tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Aspect-preserving resize then center-crop to ``(target_h, target_w)``.

    Returns ``(cropped_image, src_size, resized_size, crop_offset)``.
    """
    src_w, src_h = image.size
    scale = max(target_h / src_h, target_w / src_w)
    rw = max(target_w, int(round(src_w * scale)))
    rh = max(target_h, int(round(src_h * scale)))
    resized = image.resize((rw, rh), Image.LANCZOS)
    left = (rw - target_w) // 2
    top = (rh - target_h) // 2
    cropped = resized.crop((left, top, left + target_w, top + target_h))
    return cropped, (src_w, src_h), (rw, rh), (left, top)


# ============================================================================
# Operator
# ============================================================================


class SanaWMOperator(BaseOperator):
    """Operator for Sana-WM: image pre-processing + camera trajectory.

    Supports two interaction modes:

    1. **Action DSL**: Provide ``action_list`` (one or more ``<keys>-<frames>``
       strings, e.g. ``["w-80"]``); an ``(F, 4, 4)`` c2w trajectory is rolled out.
    2. **Explicit poses**: Provide ``c2ws`` and ``intrinsics_vec4`` as numpy arrays.
    """

    def __init__(self, operation_types: Optional[list[str]] = None):
        super().__init__(operation_types=operation_types or ["action_instruction"])
        self.interaction_template = ["prompt", "action_list", "c2ws", "intrinsics_vec4"]
        self.interaction_template_init()
        self.current_interaction = {}

    def check_interaction(self, interaction: dict) -> bool:
        if not isinstance(interaction, dict):
            raise ValueError("Interaction must be a dictionary")
        return True

    def get_interaction(self, interaction: dict) -> None:
        self.check_interaction(interaction)
        self.current_interaction = interaction

    def process_perception(
        self, input_image: Image.Image
    ) -> dict:
        """Resize + center-crop a PIL image to 704×1280 (Sana-WM's fixed resolution).

        Also returns the crop metadata needed to transform intrinsics.
        """
        cropped_img, src_size, resized_size, crop_offset = resize_and_center_crop(input_image)
        return {
            "pil_image": cropped_img,
            "src_size": src_size,
            "resized_size": resized_size,
            "crop_offset": crop_offset,
        }

    def process_interaction(
        self,
        perception: dict,
        num_frames: int = 161,
    ) -> dict:
        """Build camera trajectory and intrinsics from the current interaction.

        Args:
            perception: Output of ``process_perception`` (contains crop metadata).
            num_frames: Target number of frames (the ``predict`` step will snap it).

        Returns:
            Dict with keys ``c2ws`` ``(F, 4, 4)``, ``intrinsics_vec4`` ``(F, 4)``,
            and ``prompt``.
        """
        prompt = self.current_interaction.get("prompt", "")
        action_list = self.current_interaction.get("action_list", None)
        c2ws = self.current_interaction.get("c2ws", None)
        intrinsics_vec4 = self.current_interaction.get("intrinsics_vec4", None)

        src_size = perception.get("src_size")
        resized_size = perception.get("resized_size")
        crop_offset = perception.get("crop_offset")

        # ── Determine c2ws ─────────────────────────────────────────────
        if c2ws is not None:
            # User provided explicit poses.
            pass
        elif action_list is not None and isinstance(action_list, list):
            # Join action segments and roll out trajectory.
            action_str = ",".join(action_list)
            c2ws = action_string_to_c2w(action_str)
            # Truncate / pad to num_frames + 1 (action_string rolls N+1 poses for N frames)
            if len(c2ws) > num_frames + 1:
                c2ws = c2ws[: num_frames + 1]
        else:
            # No camera motion: identity trajectory.
            c2ws = np.tile(np.eye(4, dtype=np.float32), (num_frames + 1, 1, 1))

        # ── Determine intrinsics ────────────────────────────────────────
        if intrinsics_vec4 is None:
            # Default intrinsics: assume 35 mm equivalent on 704×1280.
            fx = fy = 1280.0
            cx = TARGET_WIDTH / 2.0
            cy = TARGET_HEIGHT / 2.0
            intrinsics_vec4 = np.array([fx, fy, cx, cy], dtype=np.float32)

        # Transform intrinsics to match the cropped image.
        if src_size and resized_size and crop_offset:
            intrinsics_vec4 = transform_intrinsics_for_crop(
                intrinsics_vec4, src_size, resized_size, crop_offset
            )

        # Broadcast to all frames if single-value.
        if intrinsics_vec4.ndim == 1:
            intrinsics_vec4 = np.broadcast_to(intrinsics_vec4, (len(c2ws), 4)).copy()

        return {
            "prompt": prompt,
            "c2ws": c2ws,
            "intrinsics_vec4": intrinsics_vec4,
        }