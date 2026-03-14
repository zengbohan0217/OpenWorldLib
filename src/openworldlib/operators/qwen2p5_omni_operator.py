"""
Qwen2.5-Omni Operator for multimodal data preprocessing.

This operator handles preprocessing for text, image, audio, and video inputs
for the Qwen2.5-Omni model.
"""

import numpy as np
from PIL import Image
import torch
from typing import Union, Optional, Dict, Any, List, Sequence
from pathlib import Path

from .base_operator import BaseOperator


class Qwen2p5OmniOperator(BaseOperator):
    """
    Operator for Qwen2.5-Omni multimodal preprocessing.
    
    Supports:
    - Text prompts
    - Image inputs (single or multiple)
    - Audio inputs (single or multiple)
    - Video inputs (with optional audio track)
    """
    
    def __init__(
        self,
        processor=None,
        use_audio_in_video: bool = True,
        system_prompt: Optional[str] = None,
        operation_types: List[str] = None,
    ):
        """
        Initialize Qwen2.5-Omni Operator
        
        Args:
            processor: Qwen2_5OmniProcessor instance
            use_audio_in_video: Whether to use audio track in video inputs
            system_prompt: System prompt for the model
            operation_types: List of operation types
        """
        if operation_types is None:
            operation_types = [
                "text_processing",
                "image_processing",
                "audio_processing",
                "video_processing",
                "multimodal_processing"
            ]
        
        super().__init__(operation_types)
        
        self.processor = processor
        self.use_audio_in_video = use_audio_in_video
        
        # Default system prompt for Qwen2.5-Omni
        if system_prompt is None:
            self.system_prompt = (
                "You are Qwen, a virtual human developed by the Qwen Team, "
                "Alibaba Group, capable of perceiving auditory and visual inputs, "
                "as well as generating text and speech."
            )
        else:
            self.system_prompt = system_prompt
        
        # Initialize interaction template
        self.interaction_template = [
            "text_prompt",
            "image_prompt",
            "audio_prompt",
            "video_prompt",
            "multimodal_prompt"
        ]
        self.interaction_template_init()
    
    def check_interaction(self, interaction):
        """Check if interaction type is valid"""
        if not isinstance(interaction, (str, dict, list)):
            raise TypeError(f"Invalid interaction type: {type(interaction)}")
        return True
    
    def get_interaction(self, interaction):
        """Get and store current interaction"""
        if self.check_interaction(interaction):
            self.current_interaction = interaction
    
    def load_image(self, image_input: Image.Image) -> Image.Image:
        """
        Load and preprocess image
        
        Args:
            image_input: PIL Image
            
        Returns:
            PIL Image in RGB mode
        """
        pil_img = image_input
        
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        return pil_img
    
    def load_audio(self, audio_input: Union[tuple, np.ndarray]) -> Union[tuple, np.ndarray]:
        """
        Load audio data
        
        Args:
            audio_input: Tuple of (numpy array, sample_rate) or numpy array
            
        Returns:
            Audio data as tuple or numpy array
        """
        # Accept preprocessed audio data (numpy array or tuple)
        if isinstance(audio_input, tuple):
            # Expecting (audio_data, sample_rate)
            return audio_input
        elif isinstance(audio_input, np.ndarray):
            return audio_input
        else:
            raise TypeError(f"Audio input must be tuple (data, sample_rate) or numpy array, got {type(audio_input)}")

    
    def load_video(self, video_input: Image.Image) -> Image.Image:
        """
        Load video frame (PIL Image)
        
        Args:
            video_input: PIL Image representing a video frame
            
        Returns:
            PIL Image
        """
        if video_input.mode != 'RGB':
            return video_input.convert('RGB')
        return video_input
    
    def process_interaction(
        self,
        text: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        include_system_prompt: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process text interaction inputs
        
        Args:
            text: Text prompt
            messages: Pre-built messages (text will be appended if provided)
            include_system_prompt: Whether to include system prompt
            **kwargs: Additional parameters
            
        Returns:
            Dict containing:
                - messages: Processed messages with text
                - text: Original text prompt
        """
        # Store current interaction
        self.get_interaction(text or messages)
        
        result = {}
        
        # Build or use provided messages
        if messages is not None:
            # Use existing messages and append text if provided
            result["messages"] = messages.copy() if isinstance(messages, list) else messages
            
            # If text is provided, append it to the last user message or create new one
            if text:
                # Find last user message
                last_user_idx = None
                for i in range(len(result["messages"]) - 1, -1, -1):
                    if result["messages"][i].get("role") == "user":
                        last_user_idx = i
                        break
                
                if last_user_idx is not None:
                    # Append to existing user message
                    if isinstance(result["messages"][last_user_idx]["content"], list):
                        result["messages"][last_user_idx]["content"].append(
                            {"type": "text", "text": text}
                        )
                    else:
                        # Convert to list format if needed
                        result["messages"][last_user_idx]["content"] = [
                            {"type": "text", "text": result["messages"][last_user_idx]["content"]},
                            {"type": "text", "text": text}
                        ]
                else:
                    # No user message found, create new one
                    result["messages"].append({
                        "role": "user",
                        "content": [{"type": "text", "text": text}]
                    })
            
            result["text"] = text
        else:
            built_messages = []
            
            # Add system prompt if requested
            if include_system_prompt and self.system_prompt:
                built_messages.append({
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt}
                    ]
                })
            
            # Build user message content with text only
            content = []
            if text:
                content.append({"type": "text", "text": text})
            
            # Add user message
            if content:
                built_messages.append({
                    "role": "user",
                    "content": content
                })
            
            result["messages"] = built_messages
            result["text"] = text
        
        return result
    
    def process_perception(
        self,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        audios: Optional[Union[tuple, np.ndarray, List]] = None,
        videos: Optional[List[Image.Image]] = None,
        include_system_prompt: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process perception inputs (images, audios, videos)
        
        Args:
            images: PIL Image or list of PIL Images
            audios: Tuple of (numpy array, sample_rate), numpy array, or list of them
            videos: List of PIL Images representing video frames
            include_system_prompt: Whether to include system prompt
            **kwargs: Additional parameters
            
        Returns:
            Dict containing:
                - messages: Processed messages with perception data
                - use_audio_in_video: Whether to use audio in video
        """
        messages = []
        
        # Add system prompt if requested
        if include_system_prompt and self.system_prompt:
            messages.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": self.system_prompt}
                ]
            })
        
        # Build user message content
        content = []
        
        # Add images
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            for img in images:
                processed_img = self.load_image(img)
                content.append({"type": "image", "image": processed_img})
        
        # Add audios
        if audios is not None:
            if not isinstance(audios, list):
                audios = [audios]
            for audio in audios:
                processed_audio = self.load_audio(audio)
                content.append({"type": "audio", "audio": processed_audio})
        
        # Add videos (as a list of PIL Images - should be added as a single video item)
        if videos is not None:
            # Videos should be a list of PIL Images representing frames
            # Convert to RGB and add as a single video item
            processed_frames = [self.load_video(frame) for frame in videos]
            content.append({"type": "video", "video": processed_frames})
        
        # Add user message
        if content:
            messages.append({
                "role": "user",
                "content": content
            })
        
        result = {
            "messages": messages,
            "use_audio_in_video": self.use_audio_in_video,
        }
        
        return result
    
    def update_config(self, **kwargs):
        """
        Update operator configuration
        
        Args:
            **kwargs: Configuration parameters to update
        """
        if "use_audio_in_video" in kwargs:
            self.use_audio_in_video = kwargs["use_audio_in_video"]
        
        if "system_prompt" in kwargs:
            self.system_prompt = kwargs["system_prompt"]
