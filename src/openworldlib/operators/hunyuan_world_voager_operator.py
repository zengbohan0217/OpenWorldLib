import numpy as np
from PIL import Image
import torch
import argparse
import os
import json
import imageio
import pyexr
import cv2

from .base_operator import BaseOperator


def camera_list(
    num_frames=49,
    type="forward",
    Width=512,
    Height=512,
    fx=256,
    fy=256,
    prev_extrinsic=None  # 上一段最后一帧的 w2c 矩阵 (4x4)，用于多轮连续运动
):
    """
    生成相机轨迹的内参和外参。
    当 prev_extrinsic 不为 None 时，从上一段轨迹的末尾位姿继续运动，
    运动方向相对于当前相机朝向。
    """
    cx = Width // 2
    cy = Height // 2
    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    intrinsics = np.stack([intrinsic] * num_frames)

    # ---- 1. 确定起始位姿 ----
    if prev_extrinsic is not None:
        R_prev = prev_extrinsic[:3, :3]
        # 从 w2c 矩阵中恢复世界坐标系下的相机位置
        start_pos   = -R_prev.T @ prev_extrinsic[:3, 3]
        # R 的每一行分别对应相机的 right / up / forward 在世界系中的方向
        cam_right   = R_prev[0, :].copy()
        cam_up      = R_prev[1, :].copy()
        cam_forward = R_prev[2, :].copy()
    else:
        start_pos   = np.array([0.0, 0.0, 0.0])
        cam_right   = np.array([1.0, 0.0, 0.0])
        cam_up      = np.array([0.0, 1.0, 0.0])
        cam_forward = np.array([0.0, 0.0, 1.0])

    # ---- 2. 根据交互类型计算终点位置 ----
    if type == "forward":
        end_pos = start_pos + cam_forward
    elif type == "backward":
        end_pos = start_pos - cam_forward
    elif type == "left":
        end_pos = start_pos - cam_right
    elif type == "right":
        end_pos = start_pos + cam_right
    else:  # camera_l, camera_r 等旋转类型，位置不变
        end_pos = start_pos.copy()

    # 插值相机中心位置
    camera_centers = np.linspace(start_pos, end_pos, num_frames)

    # ---- 3. 计算注视目标点 ----
    if type == "camera_l":
        target_start = start_pos + cam_forward * 100
        target_end   = start_pos - cam_right * 100
        target_points = np.linspace(target_start, target_end, num_frames * 2)[:num_frames]
    elif type == "camera_r":
        target_start = start_pos + cam_forward * 100
        target_end   = start_pos + cam_right * 100
        target_points = np.linspace(target_start, target_end, num_frames * 2)[:num_frames]
    else:
        # 平移运动：注视方向保持 cam_forward 不变
        # 每帧的注视点 = 当前位置 + cam_forward * 100
        target_points = camera_centers + cam_forward[np.newaxis, :] * 100

    # ---- 4. 构建每帧的 extrinsic (w2c) 矩阵 ----
    extrinsics = []
    for t, target_point in zip(camera_centers, target_points):
        z = target_point - t
        z = z / np.linalg.norm(z)

        # 用 cam_right 作为参考向量构建正交坐标系（与原代码 x=[1,0,0] 的逻辑一致）
        y = np.cross(z, cam_right)
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-6:
            # z 与 cam_right 近乎平行（极端旋转），改用 cam_up 作为备选
            y = cam_up.copy()
            norm_y = np.linalg.norm(y)
        y = y / norm_y
        x = np.cross(y, z)
        x = x / np.linalg.norm(x)

        R = np.stack([x, y, z], axis=0)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = -R @ t
        extrinsics.append(w2c)

    extrinsics = np.stack(extrinsics)
    return intrinsics, extrinsics


class HunyuanWorldVoyagerOperator(BaseOperator):
    def __init__(self, 
                 operation_types=["action_instruction"],
                 interaction_template = ["forward", "backward", "left", "right", "camera_l", "camera_r"]
        ):
        super(HunyuanWorldVoyagerOperator, self).__init__()
        self.interaction_template = interaction_template
        self.interaction_template_init()

        self.opration_types = operation_types

    def check_interaction(self, interaction):
        if interaction not in self.interaction_template:
            raise ValueError(f"Interaction {interaction} not in interaction_template")
        return True
    
    def get_interaction(self, interaction):
        """支持单个交互或交互序列"""
        if isinstance(interaction, str):
            # 单个交互
            if self.check_interaction(interaction):
                self.current_interaction.append(interaction)
        elif isinstance(interaction, list):
            # 交互序列
            for inter in interaction:
                if self.check_interaction(inter):
                    self.current_interaction.append(inter)
        else:
            raise ValueError(f"Interaction must be a string or list, got {type(interaction)}")

    def process_interaction(self,
                            num_frames,
                            Width=512,
                            Height=512,
                            fx=256,
                            fy=256
                            ):
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return camera_list(
            num_frames=num_frames,
            type=now_interaction,
            Width=Width,
            Height=Height,
            fx=fx,
            fy=fy)
    
    def process_interaction_sequence(self,
                                    interaction_sequence,
                                    num_frames_per_interaction,
                                    Width=512,
                                    Height=512,
                                    fx=256,
                                    fy=256
                                    ):
        """处理交互序列，返回所有帧的相机参数
        
        Args:
            interaction_sequence: 交互序列，如 ["forward", "backward", "camera_l"]
            num_frames_per_interaction: 每个交互的帧数
            Width, Height, fx, fy: 相机参数
            
        Returns:
            intrinsics, extrinsics: 所有帧的相机内参和外参
        """
        all_intrinsics = []
        all_extrinsics = []
        
        for interaction in interaction_sequence:
            if self.check_interaction(interaction):
                self.interaction_history.append(interaction)
                intrinsics, extrinsics = camera_list(
                    num_frames=num_frames_per_interaction,
                    type=interaction,
                    Width=Width,
                    Height=Height,
                    fx=fx,
                    fy=fy
                )
                all_intrinsics.append(intrinsics)
                all_extrinsics.append(extrinsics)
        
        # 拼接所有帧的相机参数
        if len(all_intrinsics) > 0:
            all_intrinsics = np.concatenate(all_intrinsics, axis=0)
            all_extrinsics = np.concatenate(all_extrinsics, axis=0)
        else:
            raise ValueError("No valid interactions in sequence")
            
        return all_intrinsics, all_extrinsics

    def process_perception(self, input_image, device):
        if isinstance(input_image, np.ndarray):
            image_tensor = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        elif isinstance(input_image, Image.Image):
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
            input_image = np.array(input_image)
            image_tensor = torch.tensor(input_image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        else:
            image_tensor = input_image.to(device)
        return input_image, image_tensor
