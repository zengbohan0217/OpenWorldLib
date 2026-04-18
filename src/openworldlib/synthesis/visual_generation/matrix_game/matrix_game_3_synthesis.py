import os
import sys
import logging
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List, Any, Dict

import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from einops import rearrange
from diffusers.utils import export_to_video

from ...base_synthesis import BaseSynthesis
from .matrix_game_3.utils.visualize import process_video


class MatrixGame3Synthesis(BaseSynthesis):
    """
    Synthesis module for Matrix-Game-3.

    We load the upstream model once in `from_pretrained` and reuse it.
    """

    def __init__(
        self,
        pipeline: Any,
        checkpoint_dir: str,
        code_dir: str,
        device: str = "cuda",
    ):
        super().__init__()
        self.pipeline = pipeline
        self.checkpoint_dir = checkpoint_dir
        self.code_dir = code_dir
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device: str = "cuda",
        code_dir: Optional[str] = None,
        visualize_warning: bool = False,
        **kwargs,
    ) -> "MatrixGame3Synthesis":
        if not pretrained_model_path:
            raise ValueError("MatrixGame3Synthesis requires `pretrained_model_path`.")

        if not visualize_warning:
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated\. Please use `torch\.amp\.autocast\('cuda', args\.\.\.\)` instead\.",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*torch\.load.*weights_only=False.*",
                category=FutureWarning,
            )
            logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
            logging.getLogger("torch._inductor.autotune_process").setLevel(logging.WARNING)
            logging.getLogger("torch._inductor").setLevel(logging.WARNING)
            logging.getLogger("diffusers").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)

        root = Path(__file__).resolve().parents[5]
        default_code_dir = (
            root
            / "src"
            / "openworldlib"
            / "synthesis"
            / "visual_generation"
            / "matrix_game"
            / "matrix_game_3"
        )
        code_dir = str(code_dir or default_code_dir)
        if not Path(code_dir).exists():
            raise FileNotFoundError(
                f"Matrix-Game-3 code directory not found: {code_dir}. "
                "Please ensure MG3 code exists under "
                "`src/openworldlib/synthesis/visual_generation/matrix_game/matrix_game_3` "
                "or pass `code_dir` explicitly."
            )

        # Align with MatrixGame2 style:
        # - local path: use directly
        # - repo id: download from HuggingFace
        if os.path.isdir(pretrained_model_path):
            model_root = pretrained_model_path
        else:
            repo_name = pretrained_model_path.split("/")[-1]
            local_dir = Path.cwd() / repo_name
            local_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[MatrixGame3Synthesis] Warning: local checkpoint not found, downloading from HuggingFace repo: {pretrained_model_path}"
            )
            model_root = str(
                snapshot_download(
                    repo_id=pretrained_model_path,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                )
            )
            print(f"[MatrixGame3Synthesis] Model downloaded to: {model_root}")

        print("[MatrixGame3Synthesis] Loading model weights, please wait...", flush=True)

        # Ensure the upstream code is importable.
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)

        from .matrix_game_3.wan.configs import WAN_CONFIGS  # type: ignore
        from .matrix_game_3.pipeline.inference_pipeline import MatrixGame3Pipeline as UpstreamPipeline  # type: ignore
        from .matrix_game_3.utils import utils as mg3_utils  # type: ignore

        cfg = WAN_CONFIGS["matrix_game3"]

        # A minimal args object; upstream uses many of these fields for logging / async VAE.
        # We keep defaults that work for single GPU.
        args = SimpleNamespace(
            output_dir=str(Path.cwd() / "output" / "matrix_game_3"),
            ckpt_dir=model_root,
            visualize_warning=bool(visualize_warning),
            size="704*1280",
            save_name="matrix_game_3",
            num_iterations=12,
            use_async_vae=False,
            async_vae_warmup_iters=0,
            compile_vae=False,
            lightvae_pruning_rate=None,
            vae_type="mg_lightvae_v2",
            use_int8=False,
            verify_quant=False,
        )
        pipeline_args = args

        pipeline = UpstreamPipeline(
            config=cfg,
            checkpoint_dir=model_root,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            convert_model_dtype=False,
            args=pipeline_args,
            fa_version="0",
            use_base_model=False,
        )
        pipeline.args = pipeline_args
        pipeline._mg3_utils_module = mg3_utils

        return cls(pipeline=pipeline, checkpoint_dir=model_root, code_dir=code_dir, device=device)

    def predict(
        self,
        image: Image.Image,
        prompt: str,
        interactions: Optional[List[str]] = None,
        operator_condition: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        save_name: str = "matrix_game_3_demo",
        size: str = "704*1280",
        num_iterations: int = 12,
        num_inference_steps: int = 3,
        seed: int = 42,
        sample_shift: Optional[float] = None,
        sample_guide_scale: Optional[float] = None,
        fa_version: str = "0",
        use_int8: bool = False,
        verify_quant: bool = False,
        use_async_vae: bool = False,
        async_vae_warmup_iters: int = 0,
        compile_vae: bool = False,
        lightvae_pruning_rate: Optional[float] = None,
        vae_type: str = "mg_lightvae_v2",
        use_base_model: bool = False,
        save_video: bool = True,
        return_result: bool = False,
        visualize_warning: bool = False,
        **kwargs,
    ) -> Any:
        out_dir = Path(output_dir or (Path.cwd() / "output" / "matrix_game_3"))
        out_dir.mkdir(parents=True, exist_ok=True)

        args = getattr(self.pipeline, "args", None)
        if args is None:
            args = SimpleNamespace()
            self.pipeline.args = args

        args.output_dir = str(out_dir)
        args.ckpt_dir = str(self.checkpoint_dir)
        args.size = str(size)
        args.save_name = str(save_name)
        args.num_iterations = int(num_iterations)
        args.use_async_vae = bool(use_async_vae)
        args.async_vae_warmup_iters = int(async_vae_warmup_iters)
        args.compile_vae = bool(compile_vae)
        args.lightvae_pruning_rate = lightvae_pruning_rate
        args.vae_type = str(vae_type)
        args.use_int8 = bool(use_int8)
        args.verify_quant = bool(verify_quant)
        args.visualize_warning = bool(visualize_warning)

        self.pipeline.fa_version = str(fa_version)

        if not visualize_warning:
            logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
            logging.getLogger("torch._inductor.autotune_process").setLevel(logging.WARNING)
            logging.getLogger("torch._inductor").setLevel(logging.WARNING)

        try:
            h_str, w_str = str(size).split("*")
            max_area = int(h_str) * int(w_str)
        except Exception:
            max_area = 704 * 1280

        restore_get_data = None
        if operator_condition is not None:
            keyboard = operator_condition.get("keyboard_condition")
            mouse = operator_condition.get("mouse_condition")
            if keyboard is not None and mouse is not None:
                if not isinstance(keyboard, torch.Tensor):
                    keyboard = torch.tensor(keyboard, dtype=torch.float32)
                if not isinstance(mouse, torch.Tensor):
                    mouse = torch.tensor(mouse, dtype=torch.float32)

                keyboard = keyboard.to(dtype=torch.float32)
                mouse = mouse.to(dtype=torch.float32)

                mg3_utils = getattr(self.pipeline, "_mg3_utils_module", None)
                if mg3_utils is not None:
                    original_get_data = mg3_utils.get_data

                    def _patched_get_data(num_frames, height, width, pil_image, device=None, dtype=None):
                        input_image = torch.from_numpy(np.array(pil_image)).unsqueeze(0)
                        input_image = input_image.permute(0, 3, 1, 2)

                        def normalize_to_neg_one_to_one(x):
                            return 2.0 * x - 1.0

                        transform = mg3_utils.get_video_transform(height, width, normalize_to_neg_one_to_one)
                        input_image = transform(input_image)
                        input_image = input_image.transpose(0, 1).unsqueeze(0)

                        if keyboard.shape[0] >= num_frames:
                            kb = keyboard[:num_frames]
                            ms = mouse[:num_frames]
                        else:
                            pad = num_frames - keyboard.shape[0]
                            kb_pad = keyboard[-1:].repeat(pad, 1)
                            ms_pad = mouse[-1:].repeat(pad, 1)
                            kb = torch.cat([keyboard, kb_pad], dim=0)
                            ms = torch.cat([mouse, ms_pad], dim=0)

                        first_pose = np.concatenate([np.zeros(3), np.zeros(2)], axis=0)
                        all_poses = mg3_utils.compute_all_poses_from_actions(kb, ms, first_pose=first_pose)
                        positions = all_poses[:, :3].tolist()
                        rotations = np.concatenate(
                            [np.zeros((all_poses.shape[0], 1)), all_poses[:, 3:5]],
                            axis=1,
                        ).tolist()
                        extrinsics_all = mg3_utils.get_extrinsics(rotations, positions)
                        return (
                            input_image.to(device, dtype),
                            extrinsics_all,
                            kb.to(device, dtype).unsqueeze(0),
                            ms.to(device, dtype).unsqueeze(0),
                        )

                    mg3_utils.get_data = _patched_get_data
                    inf_mod = sys.modules.get(self.pipeline.__class__.__module__)
                    if inf_mod is not None:
                        inf_mod.get_data = _patched_get_data

                    def _restore():
                        mg3_utils.get_data = original_get_data
                        if inf_mod is not None:
                            inf_mod.get_data = original_get_data

                    restore_get_data = _restore

        try:
            result: Dict[str, Any] = self.pipeline.generate(
                prompt,
                image,
                max_area=max_area,
                shift=float(sample_shift) if sample_shift is not None else 5.0,
                num_inference_steps=int(num_inference_steps),
                guide_scale=float(sample_guide_scale) if sample_guide_scale is not None else 5.0,
                seed=int(seed),
                use_base_model=bool(use_base_model),
                args=args,
            )
        finally:
            if restore_get_data is not None:
                restore_get_data()

        video_tensor = result.get("video")
        if video_tensor is None:
            raise RuntimeError("Matrix-Game-3 did not return `video` tensor.")

        keyboard_condition = result.get("keyboard_condition")
        mouse_condition = result.get("mouse_condition")
        frame_res = result.get("frame_res", (704, 1280))

        video_path = str(out_dir / f"{save_name}.mp4")
        if save_video:
            video_np = np.ascontiguousarray(
                ((rearrange(video_tensor, "C T H W -> T H W C").float() + 1) * 127.5)
                .clip(0, 255)
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            mouse_icon = Path(self.code_dir) / "assets" / "images" / "mouse.png"
            if keyboard_condition is not None and mouse_condition is not None and mouse_icon.exists():
                process_video(
                    video_np,
                    video_path,
                    (
                        keyboard_condition.float().cpu().numpy(),
                        mouse_condition.float().cpu().numpy(),
                    ),
                    str(mouse_icon),
                    mouse_scale=0.2,
                    default_frame_res=tuple(frame_res),
                )
            else:
                # Fallback without control overlay when icon or conditions are unavailable.
                export_to_video([frame / 255.0 for frame in video_np], video_path, fps=17)

        payload = {
            "video_tensor": video_tensor,
            "video_path": video_path if save_video else None,
            "keyboard_condition": keyboard_condition,
            "mouse_condition": mouse_condition,
        }
        if return_result:
            return payload
        return payload["video_path"]
