import torch
import os
from typing import Optional, Any, Union, Dict, List
from pathlib import Path
from loguru import logger

from ...operators.mmaudio_operator import MMAudioOperator
from ...synthesis.audio_generation.mmaudio.mmaudio_synthesis import MMAudioSynthesis

class MMAudioPipeline:  
    """
    输入通过operator 处理之后再传给模型推理
    实现数据预处理和模型推理的分离
    """
    def __init__(
        self, 
        operator: Optional[MMAudioOperator] = None, 
        synthesis_model: Optional[MMAudioSynthesis] = None, 
        device: str = 'cuda'
    ):

        self.operator = operator
        self.synthesis_model = synthesis_model
        self.device = device
        
    @classmethod
    def from_pretrained(
        cls, 
        model_path: str,
        required_components: Dict[str, str],
        variant: str = "large_44k_v2",
        full_precision: bool = False,
        num_steps: int = 25,
        device: str = None, 
        logger_obj=None,
        **kwargs,
    ) -> 'MMAudioPipeline':
        """
        从预训练模型加载完整的 pipeline

        Args:
            model_path: 预训练模型路径，可以是本地路径或者hugid路径
            variant: 模型变体名称
            full_precision: 是否使用全精度（float32）
            num_steps: FlowMatching 推理步数
            device: 设备，如果为 None 则自动检测
            logger_obj: 日志记录器
            **kwargs: 额外参数
            
        Returns:
            MMAudioPipeline: 初始化的 pipeline 实例
        """
        if logger_obj:
            logger_obj.info("Loading MMAudio synthesis model...")
        
        synthesis_model = MMAudioSynthesis.from_pretrained(
            model_path=model_path,
            required_components=required_components,
            variant=variant,
            full_precision=full_precision,
            num_steps=num_steps,
            device=device,
            logger_obj=logger_obj,
            **kwargs,
        )
        
        if logger_obj:
            logger_obj.info("Initializing MMAudio operator...")
        
        operator = MMAudioOperator()

        pipeline = cls(
            operator=operator,
            synthesis_model=synthesis_model,
            device=synthesis_model.device 
        )

        if logger_obj:
            logger_obj.info("MMAudio pipeline loaded successfully")
        
        return pipeline

    def process(
        self,
        prompt: str,
        video: Optional[Union[str, Path]] = None,
        negative_prompt: str = "",
        duration: float = 8.0,
        mask_away_clip: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理输入，通过 operator 预处理后传给 synthesis 模型
        
        Args:
            prompt: 文本提示
            video: 视频文件路径（可选）
            negative_prompt: 负面提示
            duration: 音频持续时间（秒）
            mask_away_clip: 是否屏蔽 CLIP 特征
            **kwargs: 其他参数
            
        Returns:
            processed_data: 处理后的数据字典
        """
        if self.operator is None:
            raise ValueError("Operator is not initialized")
        
        processed_data:Dict[str, Any] = {}

        self.operator.get_interaction(prompt, negative_prompt)
        processed_interaction = self.operator.process_interaction()
        processed_data['prompt'] = processed_interaction['prompt']
        processed_data['negative_prompt'] = processed_interaction['negative_prompt']

        processed_perception = self.operator.process_perception(
            video=video,
            duration=duration,
            mask_away_clip=mask_away_clip,
            **kwargs
        )
        processed_data.update(processed_perception)
        
        return processed_data
    
    def __call__(
        self,
        prompt: str,
        video: Optional[Union[str, Path]] = None,
        negative_prompt: str = "",
        duration: float = 8.0,
        cfg_strength: float = 4.5,
        num_steps: int = 25,
        seed: int = 42,
        mask_away_clip: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        完整的推理流程：数据预处理 -> 模型推理
        
        Args:
            prompt: 文本提示
            video: 视频文件路径（可选）
            negative_prompt: 负面提示
            duration: 音频持续时间（秒）
            cfg_strength: Classifier-free guidance 强度
            num_steps: 生成步数
            seed: 随机种子
            mask_away_clip: 是否屏蔽 CLIP 特征
            **kwargs: 其他参数
            
        Returns:
            result: 包含音频和元数据的字典
                - audio: torch.Tensor, 生成的音频波形 [seq_len]
                - sampling_rate: int, 采样率
                - duration: float, 实际音频时长
                - prompt: str, 使用的提示
                - video_info: VideoInfo, 视频信息（如果有视频输入）
        """
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")
        
        # 1. Operator 预处理
        processed_data = self.process(
            prompt=prompt,
            video=video,
            negative_prompt=negative_prompt,
            duration=duration,
            mask_away_clip=mask_away_clip,
            **kwargs
        )
        
        # 2. Synthesis 推理
        result = self.synthesis_model.predict(
            processed_data=processed_data,
            seed=seed,
            cfg_strength=cfg_strength,
            num_steps=num_steps,
            **kwargs
        )
        
        # 3. 添加视频路径信息（如果有），方便后续保存
        if video is not None:
            result["video_path_input"] = str(video)
        
        return result

    def save_pretrained(self, save_directory: Union[str, Path]):
        """
        保存 pipeline 配置到指定目录
        
        Args:
            save_directory: 保存目录路径
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # 保存 operator 配置
        if self.operator:
            operator_config = {
                'operation_types': self.operator.opration_types if hasattr(self.operator, 'opration_types') else []
            }   
            torch.save(operator_config, save_directory / "operator_config.pt")
        
        # 保存 pipeline 配置
        pipeline_config = {
            'device': self.device,
        }
        torch.save(pipeline_config, save_directory / "pipeline_config.pt")
        
        logger.info(f"MMAudio Pipeline saved to {save_directory}")
    
    def get_operator(self) -> Optional[MMAudioOperator]:
        """获取 operator 实例"""
        return self.operator
    
    def get_synthesis_model(self) -> Optional[MMAudioSynthesis]:
        """获取 synthesis 模型实例"""
        return self.synthesis_model