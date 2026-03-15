from PIL import Image
from typing import Optional, Dict, Any
import base64
import io

from .base_operator import BaseOperator


class Wan2p5Operator(BaseOperator):
    """
    Wan2.5 数据处理 Operator
    
    负责图像编码、数据预处理等数据预处理工作
    不涉及模型推理和API调用
    """
    
    def __init__(
        self,
        operation_types: list = None
    ):
        """
        初始化 Wan25Operator
        
        Args:
            operation_types: 操作类型列表
        """
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super(Wan2p5Operator, self).__init__(operation_types)
        
        # 初始化交互模板
        self.interaction_template = ["text_prompt", "image_prompt", "multimodal_prompt"]
        self.interaction_template_init()
    
    def process_image(self, image_input: Image.Image) -> str:
        """
        编码图像为base64格式
        Args:
            image_input: PIL.Image 对象
        Returns:
            base64编码的图像字符串
        """
        if not isinstance(image_input, Image.Image):
            raise TypeError(f"image_input must be PIL.Image, got {type(image_input)}")
        
        if image_input.mode != 'RGB':
            image_input = image_input.convert('RGB')
        
        buffer = io.BytesIO()
        image_input.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        mime_type = 'image/png'
        
        encoded_string = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    
    def get_interaction(self, interaction):
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction):
        if not isinstance(interaction, str):
            raise TypeError(f"Interaction must be a string, got {type(interaction)}")
        return True

    def process_interaction(self,**kwargs) -> Dict[str, Any]:
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return {
            "processed_prompt": now_interaction
        }
    
    def process_perception(
        self,
        images: Optional[Image.Image] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理交互输入，生成模型所需的输入格式
        
        Args:
            images: 参考图像（PIL.Image，可选）
            **kwargs: 其他参数
            
        Returns:
            Dict 包含处理后的输入数据：
                - encoded_image: 编码后的图像（如果有）
                - images: 原始参考图像（如果有）
        """
        result: Dict[str, Any] = {
            "encoded_image": None,
            "images": None
        }
        
        if images is not None:
            if not isinstance(images, Image.Image):
                raise TypeError(f"images must be PIL.Image, got {type(images)}")
            result["encoded_image"] = self.process_image(images)
            result["images"] = images
        
        return result

