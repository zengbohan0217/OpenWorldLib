import inspect
import os

import torch

from ..wan.modules.vae2_2 import Wan2_2_VAE

VAE_DTYPE = torch.bfloat16


def _parse_lightvae_pruning_rate(value):
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("", "auto", "none"):
            return None
    return float(value)


def _resolve_ckpt_dir(args=None, metadata=None):
    if metadata is not None:
        ckpt_dir = metadata.get("ckpt_dir") or metadata.get("checkpoint_dir")
        if ckpt_dir:
            return str(ckpt_dir)
    if args is not None:
        ckpt_dir = getattr(args, "ckpt_dir", None) or getattr(args, "checkpoint_dir", None)
        if ckpt_dir:
            return str(ckpt_dir)
    return None


def _build_vae_paths(ckpt_dir, vae_type):
    if vae_type == "mg_lightvae":
        vae_name = "MG-LightVAE.pth"
    elif vae_type == "mg_lightvae_v2":
        vae_name = "MG-LightVAE_v2.pth"
    else:
        vae_name = "Wan2.2_VAE.pth"

    return {
        "vae_path": os.path.join(ckpt_dir, vae_name),
        "lightvae_encoder_path": os.path.join(ckpt_dir, "Wan2.2_VAE.pth"),
    }


def get_vae_config(args=None):
    if args is None:
        raise ValueError("`args` is required for get_vae_config")

    ckpt_dir = _resolve_ckpt_dir(args=args)
    if not ckpt_dir:
        raise AttributeError(
            "VAE config requires `args.ckpt_dir` (or `args.checkpoint_dir`) to locate checkpoint files."
        )

    vae_type = getattr(args, "vae_type", "mg_lightvae_v2")
    pruning_rate = _parse_lightvae_pruning_rate(getattr(args, "lightvae_pruning_rate", None))

    if pruning_rate is None:
        if vae_type == "mg_lightvae":
            pruning_rate = 0.5
        elif vae_type == "mg_lightvae_v2":
            pruning_rate = 0.75

    paths = _build_vae_paths(ckpt_dir, vae_type)
    return {
        "vae_path": paths["vae_path"],
        "vae_dtype": "bfloat16",
        "vae_type": vae_type,
        "lightvae_pruning_rate": pruning_rate,
        "lightvae_encoder_path": paths["lightvae_encoder_path"],
        "ckpt_dir": ckpt_dir,
    }


def load_vae(device_id=0, args=None, metadata=None):
    if metadata is not None:
        ckpt_dir = _resolve_ckpt_dir(metadata=metadata)
        vae_type = metadata.get("vae_type", "mg_lightvae_v2")
        pruning_rate = _parse_lightvae_pruning_rate(metadata.get("lightvae_pruning_rate", None))

        vae_path = metadata.get("vae_path")
        lightvae_encoder_path = metadata.get("lightvae_encoder_path") or metadata.get("lightvae_encoder_vae_pth")

        if not vae_path:
            if not ckpt_dir:
                raise ValueError(
                    "metadata must provide `vae_path` or `ckpt_dir/checkpoint_dir` for VAE loading"
                )
            paths = _build_vae_paths(ckpt_dir, vae_type)
            vae_path = paths["vae_path"]
            lightvae_encoder_path = lightvae_encoder_path or paths["lightvae_encoder_path"]
        elif not lightvae_encoder_path:
            lightvae_encoder_path = (
                os.path.join(os.path.dirname(vae_path), "Wan2.2_VAE.pth")
                if os.path.dirname(vae_path)
                else "Wan2.2_VAE.pth"
            )
    elif args is not None:
        config = get_vae_config(args)
        vae_path = config["vae_path"]
        pruning_rate = config["lightvae_pruning_rate"]
        lightvae_encoder_path = config["lightvae_encoder_path"]
        vae_type = config["vae_type"]
    else:
        raise ValueError("Either metadata or args must be provided")

    if pruning_rate is None:
        pruning_rate = 0.0

    # Keep legacy behavior: any mg_lightvae* type uses mg_lightvae runtime branch.
    runtime_vae_type = "wan2.2" if float(pruning_rate) <= 0.0 else "mg_lightvae"
    if isinstance(vae_type, str) and vae_type.startswith("mg_lightvae") and float(pruning_rate) > 0.0:
        runtime_vae_type = "mg_lightvae"

    device = torch.device(f"cuda:{device_id}")

    constructor_kwargs = {
        "vae_pth": vae_path,
        "device": device,
        "dtype": VAE_DTYPE,
        "vae_type": runtime_vae_type,
        "lightvae_pruning_rate": pruning_rate,
        "lightvae_encoder_vae_pth": lightvae_encoder_path,
    }
    supported_params = set(inspect.signature(Wan2_2_VAE.__init__).parameters.keys())
    constructor_kwargs = {k: v for k, v in constructor_kwargs.items() if k in supported_params}

    vae = Wan2_2_VAE(**constructor_kwargs)

    vae.model.to(device=device, dtype=VAE_DTYPE)
    if hasattr(vae, "encoder_model") and vae.encoder_model is not None:
        vae.encoder_model.to(device=device, dtype=VAE_DTYPE)
    vae.scale[0] = vae.scale[0].to(device=device, dtype=VAE_DTYPE)
    vae.scale[1] = vae.scale[1].to(device=device, dtype=VAE_DTYPE)

    return vae
