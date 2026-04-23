from __future__ import annotations

import importlib.util
import math
import re
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .constants import (
    IMAGE_TOKEN_INDEX,
    SUPPORTED_TRANSFORMERS_MAX,
    SUPPORTED_TRANSFORMERS_MIN,
    SUPPORTED_TRANSFORMERS_RANGE,
)
from .siglip_vision import SigLipImageProcessor, SigLipVisionTower


def parse_version(version: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", version)
    return tuple(int(part) for part in parts[:3])


def module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def get_package_version(package_name: str) -> str | None:
    try:
        return package_version(package_name)
    except PackageNotFoundError:
        return None


def validate_cambrian_s_environment(
    require_video: bool = False,
    module_checker: Callable[[str], bool] | None = None,
    version_getter: Callable[[str], str | None] | None = None,
) -> None:
    module_checker = module_checker or module_exists
    version_getter = version_getter or get_package_version

    required_modules = {
        "torch": "torch",
        "transformers": "transformers",
        "tokenizers": "tokenizers",
        "sentencepiece": "sentencepiece",
    }
    if require_video:
        required_modules["decord"] = "decord"

    missing = [
        package_name
        for package_name, module_name in required_modules.items()
        if not module_checker(module_name)
    ]
    if missing:
        raise RuntimeError(
            "Cambrian-S requires missing dependencies: "
            + ", ".join(missing)
            + ". Use the repository's transformers_low environment and install the missing packages."
        )

    transformers_version = version_getter("transformers")
    if not transformers_version:
        raise RuntimeError(
            "Cambrian-S requires the transformers package, but no installed version was detected."
        )

    min_version = parse_version(SUPPORTED_TRANSFORMERS_MIN)
    max_version = parse_version(SUPPORTED_TRANSFORMERS_MAX)
    current_version = parse_version(transformers_version)
    if current_version < min_version or current_version > max_version:
        raise RuntimeError(
            "Cambrian-S v1 is validated for transformers"
            f" {SUPPORTED_TRANSFORMERS_RANGE}; found {transformers_version}. "
            "Use the repository's transformers_low environment."
        )


SiglipVisionTower = SigLipVisionTower


def expand_to_square(
    image: Image.Image,
    background_color: Sequence[float] = (127, 127, 127),
) -> Image.Image:
    width, height = image.size
    if width == height:
        return image

    fill_color = tuple(int(value) for value in background_color)
    side = max(width, height)
    canvas = Image.new(image.mode, (side, side), fill_color)
    paste_x = (side - width) // 2
    paste_y = (side - height) // 2
    canvas.paste(image, (paste_x, paste_y))
    return canvas


def select_best_resolution(
    original_size: Sequence[int],
    possible_resolutions: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        scale = min(width / max(original_width, 1), height / max(original_height, 1))
        downscaled_width = int(original_width * scale)
        downscaled_height = int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if (
            effective_resolution > max_effective_resolution
            or (
                effective_resolution == max_effective_resolution
                and wasted_resolution < min_wasted_resolution
            )
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    if best_fit is None:
        raise ValueError("Cambrian-S could not determine a valid anyres resolution.")
    return best_fit


def resize_and_pad_image(
    image: Image.Image,
    target_resolution: tuple[int, int],
    background_color: Sequence[int] = (0, 0, 0),
) -> Image.Image:
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / max(original_width, 1)
    scale_h = target_height / max(original_height, 1)
    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    resized_image = image.resize((new_width, new_height))
    padded_image = Image.new("RGB", (target_width, target_height), tuple(int(value) for value in background_color))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    padded_image.paste(resized_image, (paste_x, paste_y))
    return padded_image


def divide_to_patches(image: Image.Image, patch_size: int) -> list[Image.Image]:
    patches: list[Image.Image] = []
    width, height = image.size
    for offset_y in range(0, height, patch_size):
        for offset_x in range(0, width, patch_size):
            patches.append(image.crop((offset_x, offset_y, offset_x + patch_size, offset_y + patch_size)))
    return patches


def unpad_image(features: torch.Tensor, original_size: Sequence[int]) -> torch.Tensor:
    original_w, original_h = original_size
    current_h, current_w = features.shape[1:3]

    original_aspect_ratio = original_w / max(original_h, 1)
    current_aspect_ratio = current_w / max(current_h, 1)

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_w / max(original_w, 1)
        new_height = max(int(original_h * scale_factor), 1)
        padding = max((current_h - new_height) // 2, 0)
        if padding > 0:
            return features[:, padding : current_h - padding, :, :]
        return features

    scale_factor = current_h / max(original_h, 1)
    new_width = max(int(original_w * scale_factor), 1)
    padding = max((current_w - new_width) // 2, 0)
    if padding > 0:
        return features[:, :, padding : current_w - padding, :]
    return features


def _load_pil_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _processor_background_color(processor: Any) -> tuple[int, int, int]:
    return tuple(int(value * 255) for value in getattr(processor, "image_mean", (0.5, 0.5, 0.5)))


def _processor_target_resolution(processor: Any) -> int:
    crop_size = getattr(processor, "crop_size", None)
    if isinstance(crop_size, dict):
        return int(crop_size.get("height") or crop_size.get("width") or 384)

    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        return int(
            size.get("height")
            or size.get("width")
            or size.get("shortest_edge")
            or 384
        )
    if isinstance(size, (tuple, list)):
        return int(size[0])
    return 384


def preprocess_single_image(
    image: str | Path | Image.Image,
    processor: SigLipImageProcessor | Any,
    image_aspect_ratio: str = "pad",
    anyres_max_subimages: int = 1,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    pil_image = _load_pil_image(image)
    original_size = pil_image.size
    target_resolution = _processor_target_resolution(processor)
    background_color = _processor_background_color(processor)

    if image_aspect_ratio == "anyres":
        snapshot_image = expand_to_square(pil_image, background_color=background_color).resize(
            (target_resolution, target_resolution)
        )
        possible_resolutions = [
            (int(grid_width * target_resolution), int(grid_height * target_resolution))
            for grid_width in range(1, anyres_max_subimages + 1)
            for grid_height in range(1, anyres_max_subimages + 1)
            if (grid_width * grid_height) <= anyres_max_subimages
        ]
        best_resolution = select_best_resolution(pil_image.size, possible_resolutions)
        anyres_image = resize_and_pad_image(
            pil_image,
            best_resolution,
            background_color=background_color,
        )
        patches = divide_to_patches(anyres_image, target_resolution)
        image_patches = [snapshot_image] + patches
        pixel_values = torch.stack(
            [
                processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0]
                for image_patch in image_patches
            ],
            dim=0,
        )
        anyres_grid = (
            best_resolution[1] // target_resolution,
            best_resolution[0] // target_resolution,
        )
        return pixel_values, (*original_size, *anyres_grid)

    square_image = expand_to_square(pil_image, background_color=background_color).resize(
        (target_resolution, target_resolution)
    )
    pixel_values = processor.preprocess(square_image, return_tensors="pt")["pixel_values"]
    return pixel_values, original_size


def _get_model_video_attr(model_config: Any, attr_name: str, default: Any) -> Any:
    if model_config is None:
        return default
    return getattr(model_config, attr_name, default)


def process_video_with_decord(
    video_file: str | Path,
    model_config: Any = None,
    max_frames: int | None = None,
    num_threads: int = -1,
) -> tuple[np.ndarray, float, str, int]:
    validate_cambrian_s_environment(require_video=True)
    from decord import VideoReader, cpu

    if num_threads < 1:
        reader = VideoReader(str(video_file), ctx=cpu(0))
    else:
        reader = VideoReader(str(video_file), ctx=cpu(0), num_threads=num_threads)

    total_frame_num = len(reader)
    video_time = total_frame_num / reader.get_avg_fps()
    avg_fps = round(reader.get_avg_fps() / _get_model_video_attr(model_config, "video_fps", 1))
    frame_idx = [frame for frame in range(0, total_frame_num, avg_fps)]
    frame_time = [frame / avg_fps for frame in frame_idx]

    video_max_frames = (
        max_frames
        if max_frames is not None
        else _get_model_video_attr(model_config, "video_max_frames", 32)
    )
    video_force_sample = _get_model_video_attr(model_config, "video_force_sample", False)
    if video_max_frames > 0:
        if len(frame_idx) > video_max_frames or video_force_sample:
            uniform_sampled_frames = np.linspace(
                0,
                total_frame_num - 1,
                video_max_frames,
                dtype=int,
            )
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [frame / reader.get_avg_fps() for frame in frame_idx]

    video = reader.get_batch(frame_idx).asnumpy()
    frame_time_str = ",".join([f"{timestamp:.2f}s" for timestamp in frame_time])
    num_frames_to_sample = len(frame_idx)
    reader.seek(0)
    return video, video_time, frame_time_str, num_frames_to_sample


def sample_video_frames(
    video_path: str | Path,
    max_frames: int = 8,
    model_config: Any = None,
    num_threads: int = -1,
) -> list[Image.Image]:
    if model_config is None:
        model_config = type(
            "CambrianVideoConfig",
            (),
            {
                "video_fps": 1,
                "video_max_frames": max_frames,
                "video_force_sample": False,
            },
        )()
    video, _, _, _ = process_video_with_decord(
        video_path,
        model_config=model_config,
        num_threads=num_threads,
    )
    return [Image.fromarray(frame).convert("RGB") for frame in video]


def preprocess_video_frames(
    frames: Sequence[Image.Image] | str | Path,
    processor: SigLipImageProcessor | Any,
    max_frames: int | None = None,
    model_config: Any = None,
    num_threads: int = -1,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    if isinstance(frames, (str, Path)):
        raw_video, _, _, _ = process_video_with_decord(
            frames,
            model_config=model_config,
            max_frames=max_frames,
            num_threads=num_threads,
        )
        frames = [Image.fromarray(frame).convert("RGB") for frame in raw_video]
    else:
        frames = [_load_pil_image(frame) for frame in frames]

    if not frames:
        raise ValueError("Cambrian-S received an empty video input.")

    original_size = frames[0].size
    square_frames = [
        expand_to_square(frame, background_color=_processor_background_color(processor))
        for frame in frames
    ]
    pixel_values = processor.preprocess(square_frames, return_tensors="pt")["pixel_values"]
    return pixel_values, (original_size[0], original_size[1], len(frames))


def tokenizer_image_token(
    prompt: str,
    tokenizer: Any,
    image_token_index: int = IMAGE_TOKEN_INDEX,
    return_tensors: str | None = None,
) -> list[int] | torch.Tensor:
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    input_ids: list[int] = []
    offset = 0
    if prompt_chunks and prompt_chunks[0] and prompt_chunks[0][0] == getattr(tokenizer, "bos_token_id", None):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for chunk_index, chunk_ids in enumerate(prompt_chunks):
        input_ids.extend(chunk_ids[offset:])
        if chunk_index < len(prompt_chunks) - 1:
            input_ids.append(image_token_index)

    if return_tensors is None:
        return input_ids
    if return_tensors != "pt":
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return torch.tensor(input_ids, dtype=torch.long)


def resize_patch_grid(features: torch.Tensor, target_side: int) -> torch.Tensor:
    token_count = features.shape[1]
    current_side = int(token_count ** 0.5)
    if current_side * current_side != token_count and (token_count - 1) > 0:
        maybe_square = int((token_count - 1) ** 0.5)
        if maybe_square * maybe_square == (token_count - 1):
            features = features[:, 1:, :]
            token_count = features.shape[1]
            current_side = int(token_count ** 0.5)

    if current_side == target_side:
        return features

    batch, _, channels = features.shape
    grid = features.view(batch, current_side, current_side, channels).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=(target_side, target_side), mode="bilinear", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(batch, target_side * target_side, channels).to(features.dtype)


__all__ = [
    "SigLipImageProcessor",
    "SigLipVisionTower",
    "SiglipVisionTower",
    "divide_to_patches",
    "expand_to_square",
    "process_video_with_decord",
    "preprocess_single_image",
    "preprocess_video_frames",
    "resize_and_pad_image",
    "resize_patch_grid",
    "sample_video_frames",
    "select_best_resolution",
    "tokenizer_image_token",
    "unpad_image",
    "validate_cambrian_s_environment",
]
