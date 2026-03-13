import os
import torch
from pathlib import Path
from typing import Optional, Union, Dict, Any
from loguru import logger

from ...operators.thinksound_operator import ThinkSoundOperator
from ...synthesis.audio_generation.thinksound.thinksound_synthesis import ThinkSoundSynthesis


class ThinkSoundPipeline:
    """
    ThinkSound Pipeline
    
    对外暴露 “视频 + 文本 -> 音频” 的统一接口
    """
    
    def __init__(
        self, 
        operator: Optional[ThinkSoundOperator] = None,
        synthesis_model: Optional[ThinkSoundSynthesis] = None, 
        device: str = 'cuda'
    ):
        """
        初始化 ThinkSoundPipeline
        
        Args:
            operator: ThinkSoundOperator 实例
            synthesis_model: ThinkSoundSynthesis 实例
            device: 设备
        """
        self.operator = operator
        self.synthesis_model = synthesis_model
        self.device = device
    
    @classmethod
    def from_pretrained(
        cls, 
        model_path: str,
        required_components: Optional[Dict[str, str]] = None,
        model_config: str = "src/openworldlib/synthesis/audio_generation/thinksound/ThinkSound/ThinkSound/configs/model_configs/thinksound.json",
        duration_sec: float = 8.0,
        seed: int = 42,
        compile: bool = False,
        video_dir: str = "videos",
        cot_dir: str = "cot_coarse",
        results_dir: str = "results",
        scripts_dir: str = ".",
        synchformer_ckpt_path: str = "hugid/synchformer_state_dict.pth",
        device: str = None, 
        logger_obj=None,
        **kwargs
    ) -> 'ThinkSoundPipeline':
        """
        从预训练模型加载完整的 pipeline
        
        Args:
            model_path: 模型根目录或 HuggingFace repo_id
            required_components: 额外依赖组件字典，目前支持：
                - "clip_backbone_id": MetaCLIP 模型 ID 或本地路径
                - "t5_model_id": T5 模型 ID 或本地路径
                - "clip_processor_id": CLIP Processor 模型 ID 或本地路径
            model_config: 模型配置 json 路径
            duration_sec: 音频时长（秒）
            seed: 随机种子
            compile: 是否对模型进行 torch.compile
            video_dir: 视频目录
            cot_dir: COT 目录
            results_dir: 结果目录
            scripts_dir: 脚本目录
            synchformer_ckpt_path: Synchformer 模型路径
            device: 设备
            logger_obj: 日志记录器
            
        Returns:
            ThinkSoundPipeline 实例
        """
        if required_components is None:
            required_components = {
                "clip_backbone_id": "facebook/metaclip-h14-fullcc2.5b",
                "t5_model_id": "google/t5-v1_1-xl",
                "clip_processor_id": "openai/clip-vit-large-patch14",
            }
        else:
            required_components = dict(required_components)
            required_components.setdefault("clip_backbone_id", "facebook/metaclip-h14-fullcc2.5b")
            required_components.setdefault("t5_model_id", "google/t5-v1_1-xl")
            required_components.setdefault("clip_processor_id", "openai/clip-vit-large-patch14")

        if logger_obj:
            logger_obj.info("Loading ThinkSound pipeline...")
        
        if logger_obj:
            logger_obj.info("Loading ThinkSound synthesis model...")
        
        synthesis_model = ThinkSoundSynthesis.from_pretrained(
            model_path=model_path,
            model_config=model_config,
            duration_sec=duration_sec,
            seed=seed,
            compile=compile,
            device=device,
            logger_obj=logger_obj,
            **kwargs
        )
        
        operator = ThinkSoundOperator(
            video_dir=video_dir,
            cot_dir=cot_dir,
            results_dir=results_dir,
            scripts_dir=scripts_dir,
            synchformer_ckpt_path=synchformer_ckpt_path,
            required_components=required_components,
        )
        
        pipeline = cls(
            operator=operator,
            synthesis_model=synthesis_model,
            device=synthesis_model.device
        )
        
        if logger_obj:
            logger_obj.info("ThinkSound pipeline loaded successfully")
        
        return pipeline
    
    def process(
        self,
        video_path: Union[str, Path],
        title: str,
        description: str,
        use_half: bool = False,
        **kwargs
    ) -> Dict[str, Any]:

        if self.operator is None:
            raise ValueError("Operator is not initialized")

        processed_data:Dict[str, Any] = {}
        

        self.operator.get_interaction(title, description)
        self.operator.process_interaction()
        
        processed_data = self.operator.process_perception(
            video_path=video_path,
            use_half=use_half,
            device=self.device,
            **kwargs
        )
        
        return processed_data
    
    def __call__(
        self,
        video_path: Union[str, Path],
        title: str,
        description: str,
        use_half: bool = False,
        cfg_scale: float = 5.0,
        num_steps: int = 24,
        **kwargs
    ) -> Dict[str, Any]:
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")
        
        processed_data = self.process(
            video_path=video_path,
            title=title,
            description=description,
            use_half=use_half,
            **kwargs
        )
        
        result = self.synthesis_model.predict(
            processed_data=processed_data,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            **kwargs
        )
        
        result["video_path"] = str(video_path)
        
        return result
    
    def save_pretrained(self, save_directory: Union[str, Path]):
        """
        保存 pipeline 配置到指定目录
        
        Args:
            save_directory: 保存目录路径
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        if self.synthesis_model:
            import json
            config = {
                'model_config': getattr(self.synthesis_model, "model_config_path", None),
                'ckpt_dir': getattr(self.synthesis_model, "ckpt_path", None),
                'pretransform_ckpt_path': getattr(self.synthesis_model, "pretransform_ckpt_path", None),
                'duration_sec': getattr(self.synthesis_model, "duration_sec", None),
                'seed': getattr(self.synthesis_model, "seed", None),
                'compile': getattr(self.synthesis_model, "compile_model", None),
            }
            with open(save_directory / "thinksound_config.json", 'w') as f:
                json.dump(config, f, indent=2)
        
        logger.info(f"ThinkSound Pipeline saved to {save_directory}")
    
    def get_operator(self) -> Optional[ThinkSoundOperator]:
        """获取 operator 实例"""
        return self.operator
    
    def get_synthesis_model(self) -> Optional[ThinkSoundSynthesis]:
        """获取 synthesis 模型实例"""
        return self.synthesis_model
    
