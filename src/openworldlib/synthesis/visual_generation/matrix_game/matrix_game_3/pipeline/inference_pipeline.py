import gc
import json
import logging
import math
import os
import random
import sys
import types
import numpy as np
import torch
import atexit
import time
import torch.distributed as dist

from PIL import Image
from tqdm import tqdm
from einops import rearrange
from functools import partial

from ......base_models.diffusion_model.video.wan_2p2.distributed.fsdp import shard_model
from ......base_models.diffusion_model.video.wan_2p2.distributed.sequence_parallel import (
    sp_attn_forward,
    sp_dit_forward,
)
from ......base_models.diffusion_model.video.wan_2p2.distributed.util import get_world_size
from ..wan.modules.model import WanModel
from ......base_models.diffusion_model.video.wan_2p1.modules.t5 import T5EncoderModel
from ......base_models.diffusion_model.video.wan_2p2.modules.vae2_2 import Wan2_2_VAE
from ......base_models.diffusion_model.video.wan_2p1.utils.fm_solvers_unipc import (
    FlowUniPCMultistepScheduler,
)

from ..utils.cam_utils import (
    compute_relative_poses,
    select_memory_idx_fov,
    get_intrinsics,
    _interpolate_camera_poses_handedness,
)
from ..utils.utils import get_data, build_plucker_from_c2ws, build_plucker_from_pose
from .vae_worker import start_vae_worker_process


class MatrixGame3Pipeline:
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        use_camera=False,
        args=None,
        fa_version=None,
        use_base_model=False,
    ):
        r"""
        Initializes the matrix-game 3.0 generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu
        self.fa_version = fa_version

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        
        t5_ckpt_path = os.path.join(checkpoint_dir, config.t5_checkpoint)
        t5_tokenizer_path = os.path.join(checkpoint_dir, config.t5_tokenizer)

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=t5_ckpt_path,
            tokenizer_path=t5_tokenizer_path,
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size

        self.use_async_vae = getattr(args, 'use_async_vae', False)
        if self.use_async_vae:
            raise NotImplementedError(
                "use_async_vae=True is currently unsupported for tensor-first output mode. "
                "Please set use_async_vae=False."
            )

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.weight_dtype = torch.bfloat16

        if use_base_model:
            model_ckpt_dir = os.path.join(checkpoint_dir, "base_model")
        else:
            model_ckpt_dir = os.path.join(checkpoint_dir, "base_distilled_model")

        model_config = WanModel.load_config(model_ckpt_dir)
        if isinstance(model_config, tuple):
            model_config = model_config[0]
                      
        if dist.is_initialized():
            dist.barrier()

        logging.info("Initializing Model (DiT)...")
        self.model = WanModel.from_pretrained(
            model_ckpt_dir, 
            torch_dtype=torch.bfloat16,
            **model_config)
        logging.info("Model initialized.")

        if args is not None and getattr(args, 'use_int8', False):
            use_int8 = True
        elif getattr(config, 'use_int8', False):
            use_int8 = True
        else:
            use_int8 = False
        if use_int8:
            raise NotImplementedError(
                "Int8 quantization for Matrix-Game-3 is not yet wired to base_models/wan_2p2. "
                "Please run with use_int8=False."
            )

        if dist.is_initialized():
             dist.barrier()
            
        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype,)

        if self.rank == 0:
            from .vae_config import load_vae, get_vae_config
            self.vae = load_vae(device_id=device_id, args=args)
            self.vae_config_dict = get_vae_config(args)
        else:
            self.vae = None
            self.vae_config_dict = None
        
        if not t5_fsdp:
            self.text_encoder.model.to(self.device)

        self.model.to(dtype=torch.bfloat16, device=self.device)
        self.model.requires_grad_(False)
        self.model.eval()

        if self.rank == 0:
            self.output_dir = args.output_dir
            if self.use_async_vae:
                worker_gpu_id = int(get_world_size())
                if self.rank == 0:
                    meta = {
                        "height": 704,
                        "width": 1280,
                        "num_iterations": args.num_iterations if hasattr(args, 'num_iterations') else 12,
                        "compile_vae": getattr(args, 'compile_vae', False),
                        "async_vae_warmup_iters": getattr(args, 'async_vae_warmup_iters', 0),
                    }
                    meta.update(self.vae_config_dict)
                    (
                        self.vae_process,
                        self.vae_latent_queue,
                        self.vae_ack_queue,
                        self.vae_done_event,
                    ) = start_vae_worker_process(self.output_dir, worker_gpu_id, meta)
                    if self.rank == 0 and getattr(args, "visualize_warning", False):
                        print(f"\n[Rank 0] Async VAE Worker started on GPU {worker_gpu_id} (PID: {self.vae_process.pid})\n", flush=True)
            else:
                if self.rank == 0 and getattr(args, "visualize_warning", False):
                    print(f"\n[Rank 0] Running in SERIAL mode (Diffusion and VAE interleaved)\n", flush=True)

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _log_flash_attention_config(self, args):
        if self.rank != 0:
            return

        from ......base_models.diffusion_model.video.wan_2p1.modules.attention import (
            FLASH_ATTN_2_AVAILABLE,
            FLASH_ATTN_3_AVAILABLE,
        )

        requested_fa = getattr(args, 'fa_version', None)
        actual_fa = "None (SDPA)"
        if requested_fa == '0':
            actual_fa = "Disabled (SDPA)"
        elif (requested_fa == '3' or requested_fa is None) and FLASH_ATTN_3_AVAILABLE:
            actual_fa = "Flash Attention 3"
        elif FLASH_ATTN_2_AVAILABLE:
            actual_fa = "Flash Attention 2"
            if requested_fa == '3':
                print(
                    "⚠️  WARNING: Flash Attention 3 requested but not available. "
                    "Falling back to Flash Attention 2.",
                    flush=True,
                )

        if self.rank == 0 and getattr(args, "visualize_warning", False):
            print("🚀 Flash Attention Configuration:", flush=True)
            print(f"  Requested: {requested_fa if requested_fa else 'Default (3)'}", flush=True)
            print(f"  Actual:    {actual_fa}", flush=True)


    def generate(self,
                 text,
                 pil_image,
                 max_area=704 * 1280,
                 shift=5.0,
                 num_inference_steps=40,
                 guide_scale=5.0,
                 seed=-1,
                 use_base_model=False,
                 args=None):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            num_inference_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
        """
        self._log_flash_attention_config(args)
        vae_cache = [None for _ in range(32)]
        weight_dtype = torch.bfloat16
        height = int(args.size.split("*")[0])
        width = int(args.size.split("*")[1])
        save_name = args.save_name

        if self.use_async_vae:
            raise NotImplementedError(
                "use_async_vae=True is currently unsupported for tensor-first output mode. "
                "Please set use_async_vae=False."
            )

        clip_frame = 56
        first_clip_frame = clip_frame + 1
        past_frame = 16

        num_iterations = args.num_iterations

        generator = torch.Generator(device=self.device).manual_seed(seed)
        num_frames = first_clip_frame + (num_iterations - 1) * 40

        current_image, extrinsics_all, keyboard_condition_all, mouse_condition_all = get_data(num_frames, height, width, pil_image, device=self.device, dtype=weight_dtype)
        cond = self.text_encoder([text], device = self.device)
        neg_cond = self.text_encoder([self.config.sample_neg_prompt], device = self.device)

        h_orig = current_image.shape[-2]
        w_orig = current_image.shape[-1]
        aspect_ratio = h_orig / w_orig
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        target_h = lat_h * self.vae_stride[1]
        target_w = lat_w * self.vae_stride[2]

        base_K = get_intrinsics(target_h, target_w)

        if self.rank == 0:
            img_cond = self.vae.encode([current_image[0]])[0].unsqueeze(0).to(device=self.device, dtype=weight_dtype).contiguous()
        else:
            img_cond = torch.zeros((1, 48, 1, lat_h, lat_w), device=self.device, dtype=weight_dtype).contiguous()
        if dist.is_initialized():
            dist.broadcast(img_cond, src=0)

        max_lat_f = (first_clip_frame - 1) // self.vae_stride[0] + 1
        max_mem_f = 5
        max_total_f = max_lat_f + max_mem_f
        max_seq_len = max_total_f * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])

        if self.sp_size > 1:
            max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        with torch.no_grad():
            total_frames = 0
            all_latents_list = []
            all_videos_list = []

            for clip_idx in range(num_iterations):
                first_clip = (clip_idx == 0)
                if self.rank == 0 and getattr(args, "visualize_warning", False):
                    print(f" Iteration {clip_idx + 1}/{num_iterations}", flush=True)

                def align_frame_to_block(frame_idx):
                    return (frame_idx - 1) // 4 * 4 + 1 if frame_idx > 0 else 1

                def get_latent_idx(frame_idx):
                    return (frame_idx - 1) // 4 + 1

                current_end_frame_idx = first_clip_frame if first_clip else first_clip_frame + clip_idx * (clip_frame - past_frame)
                current_start_frame_idx = 0 if first_clip else current_end_frame_idx - clip_frame

                c2ws_chunk = extrinsics_all[current_start_frame_idx:current_end_frame_idx]
                src_indices = np.linspace(current_start_frame_idx, current_end_frame_idx - 1, first_clip_frame if first_clip else clip_frame)
                tgt_len = (first_clip_frame - 1) // 4 + 1 if first_clip else (clip_frame // 4)
                tgt_indices = np.linspace(0 if first_clip else current_start_frame_idx + 3, current_end_frame_idx - 1, tgt_len)
                c2ws_chunk_gpu = c2ws_chunk.to(device=self.device)

                plucker = build_plucker_from_c2ws(
                    c2ws_chunk_gpu,
                    src_indices,
                    tgt_indices,
                    framewise=True,
                    base_K=base_K,
                    target_h=target_h,
                    target_w=target_w,
                    lat_h=lat_h,
                    lat_w=lat_w,
                )
                plucker_no_mem = plucker

                if first_clip:
                    x_memory = None
                    memory_mouse_condition = None
                    memory_keyboard_condition = None
                    latent_idx = None
                    timestep_memory = None
                else:                   
                    if self.rank == 0:
                        mem_end = ((current_start_frame_idx - 1) // 4 * 4 + 1) if current_start_frame_idx > 1 else 1
                        selected_index_base = [current_end_frame_idx - o for o in range(1, 34, 8)]
                        selected_index = select_memory_idx_fov(
                            extrinsics_all,
                            current_start_frame_idx,
                            selected_index_base,
                            use_gpu=True
                        )
                        selected_index[-1] = 4 
                        selected_index_base = [current_end_frame_idx - o for o in range(1, 34, 8)]
                    else:
                        selected_index = [0] * 5 
                        selected_index_base = [current_end_frame_idx - o for o in range(1, 34, 8)]

                    if dist.is_initialized():
                        dist.broadcast_object_list(selected_index, src=0)

                    memory_pluckers = []
                    latent_idx = []
                    for mem_idx, reference_idx in zip(selected_index, selected_index_base):

                        l_idx = get_latent_idx(mem_idx)
                        latent_idx.append(l_idx)

                        mem_idx_aligned = align_frame_to_block(mem_idx)
                        mem_block = extrinsics_all[mem_idx_aligned:mem_idx_aligned + 4]
                        mem_src = np.linspace(mem_idx_aligned, mem_idx_aligned + 3, mem_block.shape[0])
                        mem_tgt = np.array([mem_idx_aligned + 3], dtype=np.float32)
                        mem_pose = _interpolate_camera_poses_handedness(
                            src_indices=mem_src,
                            src_rot_mat=mem_block[:, :3, :3].cpu().numpy(),
                            src_trans_vec=mem_block[:, :3, 3].cpu().numpy(),
                            tgt_indices=mem_tgt,
                        )
                        reference_pose = extrinsics_all[reference_idx:reference_idx + 1]
                        rel_pair = torch.cat([reference_pose, mem_pose], dim=0)
                        rel_pose = compute_relative_poses(rel_pair, framewise=False)[1:2]
                        rel_pose_gpu = rel_pose.to(device=self.device)

                        memory_pluckers.append(
                            build_plucker_from_pose(
                                rel_pose_gpu,
                                base_K=base_K,
                                target_h=target_h,
                                target_w=target_w,
                                lat_h=lat_h,
                                lat_w=lat_w,
                            )
                        )
                    plucker = torch.cat(memory_pluckers + [plucker], dim=2)
                    src = torch.cat(all_latents_list, dim=2)
                    x_memory = src[:, :, latent_idx]
                    memory_mouse_condition = torch.ones((1, len(selected_index), 2)).to(device=self.device, dtype=weight_dtype)
                    memory_keyboard_condition = -torch.ones((1, len(selected_index), 6)).to(device=self.device, dtype=weight_dtype)
                    timestep_memory = x_memory.new_zeros((1, x_memory.shape[2] * x_memory.shape[3] * x_memory.shape[4] // 4))

                keyboard_condition = keyboard_condition_all[:, current_start_frame_idx:current_end_frame_idx]
                mouse_condition = mouse_condition_all[:, current_start_frame_idx:current_end_frame_idx]
                plucker = plucker.to(device=self.device, dtype=weight_dtype)
                plucker_no_mem = plucker_no_mem.to(device=self.device, dtype=weight_dtype)

                test_scheduler = FlowUniPCMultistepScheduler()
                timesteps = test_scheduler.set_timesteps(num_inference_steps, device=self.device, shift=shift)
                timesteps = test_scheduler.timesteps
                
                latent_start_idx = get_latent_idx(current_start_frame_idx)
                latent_end_idx = get_latent_idx(current_end_frame_idx)

                latents = torch.randn((1, 48, latent_end_idx - latent_start_idx, img_cond.shape[-2], img_cond.shape[-1]), generator=generator, device=self.device, dtype=weight_dtype)
                latents = torch.cat([img_cond, latents[:, :, img_cond.shape[2]:]], dim = 2)

                conditions_full = {
                    "mouse_cond": mouse_condition,
                    "keyboard_cond": keyboard_condition,
                    "context": cond,
                    "plucker_emb": plucker,
                    "x_memory": x_memory,
                    "timestep_memory": timestep_memory,
                    "keyboard_cond_memory": memory_keyboard_condition,
                    "mouse_cond_memory": memory_mouse_condition,
                    "memory_latent_idx": latent_idx,
                    "predict_latent_idx": (latent_start_idx, latent_end_idx),
                    "fa_version": self.fa_version,
                }

                conditions_null = {
                    "mouse_cond": torch.ones_like(mouse_condition).to(device=self.device, dtype=weight_dtype),
                    "keyboard_cond": -torch.ones_like(keyboard_condition).to(device=self.device, dtype=weight_dtype),
                    "context": neg_cond,
                    "plucker_emb": plucker_no_mem,
                    "x_memory": None,
                    "timestep_memory": None,
                    "keyboard_cond_memory": None,
                    "mouse_cond_memory": None,
                    "memory_latent_idx": None,
                    "predict_latent_idx": (latent_start_idx, latent_end_idx),
                }
                    
                for _, t in enumerate(tqdm(timesteps, disable=(self.rank != 0))):
                    latent_model_input = latents

                    timestep = latents.new_full((latents.shape[2], latents.shape[3] * latents.shape[4] // 4), t)
                    timestep[:img_cond.shape[2]].zero_()
                    timestep = timestep.flatten().unsqueeze(0)
                   
                    model_kwargs = {
                        "x": latent_model_input,
                        "t": timestep,
                        "seq_len": max_seq_len,
                        **conditions_full
                    }

                    model_kwargs_null = {
                        "x": latent_model_input,
                        "t": timestep,
                        "seq_len": max_seq_len,
                        **conditions_null
                    }
                    if use_base_model:
                        noise_pred_full = self.model(**model_kwargs)
                        noise_pred_null = self.model(**model_kwargs_null)
                        noise_pred = noise_pred_null + guide_scale * (noise_pred_full - noise_pred_null)
                    else:
                        noise_pred = self.model(**model_kwargs)

                    if (
                        args is not None
                        and getattr(args, "use_int8", False)
                        and getattr(args, "verify_quant", False)
                        and getattr(args, "visualize_warning", False)
                    ):
                        if _ == 0 and self.rank == 0:
                            print(f"\n[Verification] Step {_}: noise_pred stats: mean={noise_pred.mean().item():.6f}, std={noise_pred.std().item():.6f}", flush=True)

                    latents = test_scheduler.step(
                        noise_pred, t, latents, return_dict=False)[0]
                    latents = torch.cat([img_cond, latents[:,:,img_cond.shape[2]:]], dim=2)
                                 
                img_cond = latents[:, :, -4:]
                denoised_pred = latents if first_clip else latents[:, :, -10:]

                if self.rank == 0:
                    vae_profiler = {}
                    do_compile = getattr(args, "compile_vae", False) and clip_idx >= 1
                    vae_segment_size = int(os.environ.get("WAN_VAE_SEGMENT_SIZE", "4"))
                    video, vae_cache = self.vae.stream_decode(
                        denoised_pred.to(dtype=self.vae.dtype),
                        vae_cache,
                        first_chunk=first_clip,
                        segment_size=vae_segment_size,
                        profiler=vae_profiler,
                        compile_decoder=do_compile,
                    )
                    all_videos_list.append(video.cpu())
                        
                all_latents_list.append(denoised_pred)
                current_frames = 57 if first_clip else 40
                total_frames += current_frames

            if self.rank == 0:
                if len(all_videos_list) > 0:
                    video = torch.concat(all_videos_list, dim=2)[0]
                else:
                    video = None
            else:
                video = None

            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
            return {
                "video": video,
                "video_path": None,
                "keyboard_condition": keyboard_condition_all.squeeze(0).detach().cpu() if keyboard_condition_all is not None else None,
                "mouse_condition": mouse_condition_all.squeeze(0).detach().cpu() if mouse_condition_all is not None else None,
                "frame_res": (height, width),
            }
