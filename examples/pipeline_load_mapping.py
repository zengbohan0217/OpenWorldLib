from typing import Union, Dict


def _resolve_path(model_path: Union[str, Dict], key: str) -> str:
    """
    Parse the specified path from model_path.
        - If `model_path` is a str, return it directly (backward compatibility for single-path model).
        - If `model_path` is a dict, retrieve the value by key; raise a clear error if the key does not exist.
    """
    if isinstance(model_path, dict):
        if key not in model_path:
            raise KeyError(
                f"Expected key '{key}' in model_path dict, "
                f"but only found: {list(model_path.keys())}"
            )
        return model_path[key]
    return model_path


def load_matrix_game2_pipeline(model_path: Union[str, Dict], device: str):
    from openworldlib.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline
    return MatrixGame2Pipeline.from_pretrained(
        model_path=_resolve_path(model_path, "pretrained_model_path"),
        mode="universal",
        device=device,
    )


def load_matrix_game3_pipeline(model_path: Union[str, Dict], device: str):
    from openworldlib.pipelines.matrix_game.pipeline_matrix_game_3 import MatrixGame3Pipeline
    return MatrixGame3Pipeline.from_pretrained(
        model_path=_resolve_path(model_path, "pretrained_model_path"),
        device=device,
    )


def load_hunyuan_game_craft_pipeline(model_path: Union[str, Dict], device: str):
    from openworldlib.pipelines.hunyuan_world.pipeline_hunyuan_game_craft import HunyuanGameCraftPipeline
    return HunyuanGameCraftPipeline.from_pretrained(
        model_path=_resolve_path(model_path, "pretrained_model_path"),
        device=device,
    )


def load_infinite_world_pipeline(model_path: Union[str, Dict], device: str):
    from openworldlib.pipelines.infinite_world.pipeline_infinite_world import InfiniteWorldPipeline

    required_components = None
    if isinstance(model_path, dict):
        required_components = {}
        optional_keys = {
            "checkpoint_path": "checkpoint_path",
            "vae_model_path": "vae_model_path",
            "vae_pth": "vae_pth",
            "text_encoder_model_path": "text_encoder_model_path",
            "text_encoder_checkpoint_path": "text_encoder_checkpoint_path",
            "tokenizer_path": "tokenizer_path",
        }
        for src_key, dst_key in optional_keys.items():
            value = model_path.get(src_key)
            if value is not None:
                required_components[dst_key] = value
        if len(required_components) == 0:
            required_components = None

    return InfiniteWorldPipeline.from_pretrained(
        model_path=_resolve_path(model_path, "pretrained_model_path"),
        required_components=required_components,
        device=device,
    )


def load_lingbot_world_pipeline(model_path: Union[str, Dict], device: str):
    import os
    from openworldlib.pipelines.lingbot_world.pipeline_lingbot_world import LingBotPipeline
    rank = int(os.getenv("RANK", 0))
    return LingBotPipeline.from_pretrained(
        model_path=_resolve_path(model_path, "pretrained_model_path"),
        mode="i2v-A14B",
        device=device,  
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=False,
        ulysses_size=1,
        t5_cpu=True,
        offload_model=True
    )


def load_qwen2p5_omni_pipeline(model_path: Union[str, Dict], device: str):
    from openworldlib.pipelines.qwen.pipeline_qwen2p5_omni import Qwen2p5OmniPipeline
    return Qwen2p5OmniPipeline.from_pretrained(
        pretrained_model_path=_resolve_path(model_path, "pretrained_model_path"),
        use_audio_in_video=False,
        device=device,
    )


def load_wan2p2_pipeline(model_path: Union[str, Dict], device: str):
    from openworldlib.pipelines.wan.pipeline_wan_2p2 import Wan2p2Pipeline
    return Wan2p2Pipeline.from_pretrained(
        synthesis_model_path=_resolve_path(model_path, "pretrained_model_path"),
        task="ti2v-5B",
    )


def load_spirit_v1p5_pipeline(model_path: Union[str, Dict], device: str, norm_stats_path: str = None):
    from openworldlib.pipelines.spirit_ai.pipeline_spirit_v1p5 import SpiritV1p5Pipeline
    return SpiritV1p5Pipeline.from_pretrained(
        pretrained_model_path=_resolve_path(model_path, "pretrained_model_path"),
        norm_stats_path=norm_stats_path,
        device=device,
        use_bf16=True,
    )


def load_cosmos_predict2p5_pipeline(model_path: Union[str, Dict], device: str, token: str = None, mode='img2world'):
    from openworldlib.pipelines.cosmos.pipeline_cosmos_predict2p5 import CosmosPredict2p5Pipeline
    return CosmosPredict2p5Pipeline.from_pretrained(
        model_path=_resolve_path(model_path, "pretrained_model_path"),
        required_components = {
            "text_encoder_model_path": _resolve_path(model_path, "text_encoder_model_path"),
            "vae_model_path": _resolve_path(model_path, "vae_model_path"),
        },
        token=token,
        mode=mode,
        device=device,
    )


## utilize lazy loader to load different tasks pipeline
video_gen_pipe = {
    "matrix-game2": load_matrix_game2_pipeline,
    "infinite-world": load_infinite_world_pipeline,
    "matrix-game3": load_matrix_game3_pipeline,
    "matrix-game-3": load_matrix_game3_pipeline,
    "wan2p2": load_wan2p2_pipeline,
    "hunyuan-game-craft": load_hunyuan_game_craft_pipeline,
    "lingbot-world": load_lingbot_world_pipeline,
    "cosmos-predict2p5": load_cosmos_predict2p5_pipeline,
}

reasoning_pipe = {
    "qwen2p5-omni": load_qwen2p5_omni_pipeline,
}

three_dim_pipe = {

}

vla_pipe = {
    "spirit-v1p5": load_spirit_v1p5_pipeline,
}
