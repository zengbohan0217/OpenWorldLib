import os
from typing import List, Optional, Union, Dict, Any

import numpy as np
import cv2
import torch
from PIL import Image
import json
from diffusers.utils import export_to_video

from ...operators.vggt_operator import VGGTOperator
from ...representations.point_clouds_generation.vggt.vggt_representation import (
    VGGTRepresentation,
)
from ...base_models.three_dimensions.point_clouds.gaussian_splatting.scene.dataset_readers import (
    storePly,
    fetchPly,
)
from ...representations.point_clouds_generation.flash_world.flash_world.render import (
    gaussian_render,
)


class VGGTResult:
    """Container class for VGGT inference results."""
    
    def __init__(
        self,
        images: List[Image.Image],
        numpy_data: Dict[str, np.ndarray],
        camera_params: List[Dict[str, Any]],
        data_type: str = "image"
    ):
        self.images = images
        self.numpy_data = numpy_data
        self.camera_params = camera_params
        self.data_type = data_type
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'camera_params': self.camera_params[idx] if idx < len(self.camera_params) else None,
            'numpy_data': {k: v[idx] if isinstance(v, np.ndarray) and v.ndim > len(self.images) else v 
                          for k, v in self.numpy_data.items()}
        }
    
    def save(self, output_dir: Optional[str] = None) -> List[str]:
        """Save VGGT results to files."""
        if output_dir is None:
            output_dir = "./vggt_output"
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files: List[str] = []
        
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        for i, img in enumerate(self.images):
            img_path = os.path.join(vis_dir, f"result_{i:04d}.png")
            img.save(img_path)
            saved_files.append(img_path)
        
        np_dir = os.path.join(output_dir, "numpy")
        os.makedirs(np_dir, exist_ok=True)
        for key, value in self.numpy_data.items():
            if isinstance(value, np.ndarray):
                np_path = os.path.join(np_dir, f"{key}.npy")
                np.save(np_path, value)
                saved_files.append(np_path)
        
        json_dir = os.path.join(output_dir, "json")
        os.makedirs(json_dir, exist_ok=True)
        for i, camera_param in enumerate(self.camera_params):
            json_path = os.path.join(json_dir, f"camera_{i:04d}.json")
            with open(json_path, 'w') as f:
                json.dump(camera_param, f, indent=2)
            saved_files.append(json_path)
        
        return saved_files


class VGGTPipeline:
    """Pipeline for VGGT 3D scene reconstruction."""
    
    def __init__(
        self,
        representation_model: Optional[VGGTRepresentation] = None,
        reasoning_model: Optional[Any] = None,
        synthesis_model: Optional[Any] = None,
        operator: Optional[VGGTOperator] = None,
    ) -> None:
        self.representation_model = representation_model
        self.reasoning_model = reasoning_model
        self.synthesis_model = synthesis_model
        self.operator = operator or VGGTOperator()
    
    @classmethod
    def from_pretrained(
        cls,
        representation_path: str,
        reasoning_path: Optional[str] = None,
        synthesis_path: Optional[str] = None,
        **kwargs
    ) -> 'VGGTPipeline':
        representation_model = VGGTRepresentation.from_pretrained(
            pretrained_model_path=representation_path,
            **kwargs
        )
        reasoning_model = None
        synthesis_model = None
        return cls(
            representation_model=representation_model,
            reasoning_model=reasoning_model,
            synthesis_model=synthesis_model,
        )
    
    def process(
        self,
        input_: Union[str, np.ndarray, List[str], List[np.ndarray]],
        interaction: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> VGGTResult:
        if self.representation_model is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")

        images_data = self.operator.process_perception(input_)
        if not isinstance(images_data, list):
            images_data = [images_data]
        
        if interaction is None:
            interaction_dict = {
                'predict_cameras': True,
                'predict_depth': True,
                'predict_points': True,
                'predict_tracks': False,
            }
        elif isinstance(interaction, str):
            self.operator.get_interaction(interaction)
            interaction_dict = self.operator.process_interaction()
        else:
            interaction_dict = interaction
        
        data = {
            'images': images_data,
            'predict_cameras': interaction_dict.get('predict_cameras', True),
            'predict_depth': interaction_dict.get('predict_depth', True),
            'predict_points': interaction_dict.get('predict_points', True),
            'predict_tracks': interaction_dict.get('predict_tracks', False),
            'query_points': kwargs.get('query_points', None),
            'preprocess_mode': kwargs.get('preprocess_mode', 'crop'),
            'resolution': kwargs.get('resolution', 518),
        }
        
        results = self.representation_model.get_representation(data)
        
        numpy_data = {}
        for key in ['extrinsic', 'intrinsic', 'depth_map', 'depth_conf', 
                   'point_map', 'point_conf', 'point_map_from_depth',
                   'tracks', 'track_vis_score', 'track_conf_score']:
            if key in results:
                numpy_data[key] = results[key]
        
        camera_params = []
        if 'extrinsic' in results and 'intrinsic' in results:
            num_images = results['extrinsic'].shape[0] if results['extrinsic'].ndim > 2 else 1
            for i in range(num_images):
                if results['extrinsic'].ndim > 2:
                    extrinsic = results['extrinsic'][i].tolist()
                    intrinsic = results['intrinsic'][i].tolist()
                else:
                    extrinsic = results['extrinsic'].tolist()
                    intrinsic = results['intrinsic'].tolist()
                camera_params.append({
                    'extrinsic': extrinsic,
                    'intrinsic': intrinsic,
                })
        
        return_visualization = kwargs.get('return_visualization', True)
        images = []
        
        if return_visualization and 'depth_map' in results:
            depth_maps = results['depth_map']
            if depth_maps.ndim == 2:
                depth_maps = depth_maps[np.newaxis, ...]
            for i in range(depth_maps.shape[0]):
                depth = depth_maps[i]
                if depth.ndim > 2:
                    depth = depth.squeeze()
                if depth.ndim != 2:
                    raise ValueError(f"Expected 2D depth map, got shape {depth.shape}")
                depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth_uint8 = (depth_normalized * 255).astype(np.uint8)
                depth_img = Image.fromarray(depth_uint8, mode='L')
                images.append(depth_img)
        else:
            for img_data in images_data:
                if isinstance(img_data, np.ndarray):
                    img_uint8 = (img_data * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_uint8)
                    images.append(img_pil)
        
        return VGGTResult(
            images=images,
            numpy_data=numpy_data,
            camera_params=camera_params,
            data_type="image"
        )

    @staticmethod
    def _to_uint8_rgb(frame: Union[Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(frame, Image.Image):
            arr = np.array(frame.convert("RGB"))
        else:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0.0, 255.0)
                if arr.max() <= 1.0:
                    arr = arr * 255.0
                arr = arr.astype(np.uint8)
        return np.ascontiguousarray(arr[..., :3])

    def _export_video(
        self,
        frames: List[Union[Image.Image, np.ndarray]],
        output_path: str,
        fps: int = 12,
    ) -> str:
        if len(frames) == 0:
            raise RuntimeError("No frames to export.")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        frames_u8 = [self._to_uint8_rgb(f) for f in frames]

        # Some encoders/players can fail with a single-frame mp4.
        if len(frames_u8) == 1:
            frames_u8 = [frames_u8[0], frames_u8[0]]

        # Save first frame as a simple preview image.
        first_frame_path = os.path.join(os.path.dirname(output_path), "first_frame.png")
        Image.fromarray(frames_u8[0]).save(first_frame_path)

        export_to_video(frames_u8, output_path, fps=fps)
        return output_path

    @staticmethod
    def _normalize_interaction_sequence(
        interaction: Optional[Union[str, List[str]]]
    ) -> List[str]:
        if interaction is None:
            return []
        if isinstance(interaction, str):
            return [interaction]
        return [str(sig) for sig in interaction if str(sig).strip()]

    @staticmethod
    def _apply_camera_view_to_camera_cfg(
        camera_cfg: Dict[str, Any],
        camera_view: Optional[List[float]],
        camera_range: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        camera_view: [dx, dy, dz, theta_x, theta_z]
        dx,dy,dz: center offset in world space
        theta_x: pitch offset (deg)
        theta_z: yaw offset (deg)
        """
        if camera_view is None:
            return camera_cfg

        if len(camera_view) != 5:
            raise ValueError(f"camera_view must be a 5D vector [dx,dy,dz,theta_x,theta_z], got {camera_view}")

        dx, dy, dz, theta_x, theta_z = camera_view
        center = np.asarray(camera_cfg.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
        center = center + np.array([dx, dy, dz], dtype=np.float32)

        yaw = float(camera_cfg.get("yaw", 0.0) + theta_z)
        pitch = float(camera_cfg.get("pitch", 0.0) + theta_x)

        if camera_range is not None:
            yaw = max(camera_range["yaw_min"], min(camera_range["yaw_max"], yaw))
            pitch = max(camera_range["pitch_min"], min(camera_range["pitch_max"], pitch))

        camera_cfg["center"] = center.tolist()
        camera_cfg["yaw"] = yaw
        camera_cfg["pitch"] = pitch
        return camera_cfg

    @staticmethod
    def _resize_colors_to_pointmap(
        colors: List[np.ndarray],
        n_views: int,
        height: int,
        width: int,
    ) -> List[np.ndarray]:
        resized: List[np.ndarray] = []
        for i in range(n_views):
            src = np.asarray(colors[min(i, len(colors) - 1)], dtype=np.float32)
            if src.max() > 1.0:
                src = src / 255.0
            if src.shape[0] != height or src.shape[1] != width:
                src = cv2.resize(src, (width, height), interpolation=cv2.INTER_LINEAR)
            resized.append(np.clip(src, 0.0, 1.0))
        return resized

    def reconstruct_ply(
        self,
        input_: Union[str, np.ndarray, List[str], List[np.ndarray]],
        ply_path: Optional[str] = None,
        interaction: Optional[Union[str, Dict[str, Any]]] = None,
        point_conf_threshold: float = 0.2,
        resolution: int = 518,
        preprocess_mode: str = "crop",
    ) -> Dict[str, Any]:
        """
        Stage 1: reconstruct colored point cloud PLY and estimate camera range.

        Returns:
            dict with keys:
            - ply_path
            - camera_range
            - default_camera
        """
        # For VGGT, reconstruction requires cameras, depth, and points.
        # We bypass interaction strings here and directly request these predictions.
        interaction_dict: Dict[str, Any] = {
            "predict_cameras": True,
            "predict_depth": True,
            "predict_points": True,
            "predict_tracks": False,
        }

        result = self.process(
            input_=input_,
            interaction=interaction_dict,
            return_visualization=False,
            resolution=resolution,
            preprocess_mode=preprocess_mode,
        )

        if "point_map" not in result.numpy_data:
            raise RuntimeError("VGGT output does not contain point_map.")

        point_map = np.asarray(result.numpy_data["point_map"])
        if point_map.ndim == 3:
            point_map = point_map[None, ...]
        if point_map.ndim != 4 or point_map.shape[-1] != 3:
            raise RuntimeError(f"Unexpected point_map shape: {point_map.shape}")

        point_conf = result.numpy_data.get("point_conf", None)
        if point_conf is None:
            point_conf = np.ones(point_map.shape[:3], dtype=np.float32)
        else:
            point_conf = np.asarray(point_conf)
            if point_conf.ndim == 2:
                point_conf = point_conf[None, ...]

        source_colors = self.operator.process_perception(input_)
        if not isinstance(source_colors, list):
            source_colors = [source_colors]

        n_views, h, w, _ = point_map.shape
        color_maps = self._resize_colors_to_pointmap(source_colors, n_views, h, w)

        points_flat = point_map.reshape(-1, 3)
        conf_flat = point_conf.reshape(-1)
        colors_flat = np.concatenate([c.reshape(-1, 3) for c in color_maps], axis=0)

        valid = np.isfinite(points_flat).all(axis=1) & np.isfinite(colors_flat).all(axis=1)
        valid &= conf_flat >= point_conf_threshold

        points = points_flat[valid].astype(np.float32)
        colors = np.clip(colors_flat[valid], 0.0, 1.0)
        if points.shape[0] == 0:
            raise RuntimeError("No valid points after confidence filtering.")

        if ply_path is None:
            output_dir = "./vggt_output"
            os.makedirs(output_dir, exist_ok=True)
            ply_path = os.path.join(output_dir, "pointcloud.ply")
        else:
            if not ply_path.endswith(".ply"):
                os.makedirs(ply_path, exist_ok=True)
                ply_path = os.path.join(ply_path, "pointcloud.ply")
            else:
                os.makedirs(os.path.dirname(ply_path) or ".", exist_ok=True)

        rgb_uint8 = (colors * 255.0).astype(np.uint8)
        storePly(ply_path, points, rgb_uint8)

        center = points.mean(axis=0)
        dists = np.linalg.norm(points - center[None, :], axis=1)
        radius = float(dists.max() + 1e-6)

        camera_range = {
            "center": center.tolist(),
            "radius_min": max(radius * 0.5, 1e-3),
            "radius_max": radius * 3.0,
            "yaw_min": -180.0,
            "yaw_max": 180.0,
            "pitch_min": -75.0,
            "pitch_max": 75.0,
        }

        # Default view distance: 1.0 = closer (was 1.5).
        default_camera = {
            "center": center.tolist(),
            "radius": radius * 1.0,
            "yaw": 0.0,
            "pitch": 0.0,
        }

        return {
            "ply_path": ply_path,
            "camera_range": camera_range,
            "default_camera": default_camera,
        }

    @staticmethod
    def _estimate_gaussian_scale(points: np.ndarray, scene_center: np.ndarray) -> float:
        if len(points) < 4:
            scene_radius = float(np.linalg.norm(points - scene_center[None, :], axis=1).max() + 1e-8)
            return max(scene_radius / 2000.0, 1e-4)

        sample_n = min(len(points), 2048)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), size=sample_n, replace=False)
        sample = torch.from_numpy(points[idx]).float()
        dist = torch.cdist(sample, sample, p=2)
        dist.fill_diagonal_(1e9)
        nn = dist.min(dim=1).values
        nn_med = float(nn.median().item())

        scene_radius = float(np.linalg.norm(points - scene_center[None, :], axis=1).max() + 1e-8)
        min_scale = max(scene_radius / 5000.0, 1e-4)
        max_scale = max(scene_radius / 300.0, min_scale)
        return float(np.clip(nn_med * 0.6, min_scale, max_scale))

    def render_with_3dgs(
        self,
        ply_path: str,
        camera_config: Dict[str, Any],
        image_width: int = 704,
        image_height: int = 480,
        device: Optional[str] = None,
        near_plane: float = 0.01,
        far_plane: float = 1000.0,
    ) -> Image.Image:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        pcd = fetchPly(ply_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32)
        if points.size == 0:
            raise RuntimeError(f"No points loaded from PLY: {ply_path}")

        scene_center = points.mean(axis=0)
        scene_radius = float(np.linalg.norm(points - scene_center[None, :], axis=1).max() + 1e-8)
        scene_radius = max(scene_radius, 1e-6)

        # Normalize scene for rendering stability across large/small VGGT scales.
        points_norm = (points - scene_center[None, :]) / scene_radius
        center = np.asarray(camera_config.get("center", scene_center.tolist()), dtype=np.float32)
        center_norm = (center - scene_center) / scene_radius

        radius_raw = float(camera_config.get("radius", 1.0 * scene_radius))
        radius_norm = max(radius_raw / scene_radius, 1e-3)
        # +180° yaw so camera is on the opposite side (scene faces camera, not back).
        yaw_deg = float(camera_config.get("yaw", 0.0)) + 180.0
        pitch_deg = float(camera_config.get("pitch", 0.0))

        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        cam_x = center_norm[0] + radius_norm * np.cos(pitch) * np.sin(yaw)
        cam_y = center_norm[1] + radius_norm * np.sin(pitch)
        cam_z = center_norm[2] + radius_norm * np.cos(pitch) * np.cos(yaw)
        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        def build_c2w(
            look_at: np.ndarray,
            eye: np.ndarray,
            reverse_forward: bool = False,
            basis_layout: str = "row",
        ) -> np.ndarray:
            forward = (eye - look_at) if reverse_forward else (look_at - eye)
            forward = forward / (np.linalg.norm(forward) + 1e-8)
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(forward, up)
            right_norm = np.linalg.norm(right)
            if right_norm < 1e-6:
                up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                right = np.cross(forward, up)
                right_norm = np.linalg.norm(right)
            right = right / (right_norm + 1e-8)
            up = np.cross(right, forward)
            up = up / (np.linalg.norm(up) + 1e-8)

            c2w_local = np.eye(4, dtype=np.float32)
            if basis_layout == "row":
                c2w_local[0, :3] = right
                c2w_local[1, :3] = up
                c2w_local[2, :3] = forward
            else:
                c2w_local[:3, 0] = right
                c2w_local[:3, 1] = up
                c2w_local[:3, 2] = forward
            c2w_local[:3, 3] = eye
            return c2w_local

        fx = 0.5 * image_width / np.tan(np.deg2rad(60.0) / 2.0)
        fy = 0.5 * image_height / np.tan(np.deg2rad(45.0) / 2.0)
        cx = image_width / 2.0
        cy = image_height / 2.0

        xyz = torch.from_numpy(points_norm).to(device=device, dtype=torch.float32)
        scale_value = self._estimate_gaussian_scale(points_norm, center_norm)
        scale = torch.full((xyz.shape[0], 3), scale_value, device=device, dtype=torch.float32)
        rotation = torch.zeros((xyz.shape[0], 4), device=device, dtype=torch.float32)
        rotation[:, 0] = 1.0
        opacity = torch.full((xyz.shape[0], 1), 0.95, device=device, dtype=torch.float32)
        color_tensor = torch.from_numpy(np.clip(colors, 0.0, 1.0)).to(device=device, dtype=torch.float32)

        gaussian_params = torch.cat([xyz, opacity, scale, rotation, color_tensor], dim=-1).unsqueeze(0)
        intr = torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32, device=device).unsqueeze(0)

        # Dynamic planes are more robust for arbitrary VGGT world scales.
        near_dynamic = max(near_plane, radius_norm * 0.01)
        far_dynamic = max(far_plane, radius_norm * 20.0)

        if not hasattr(self, "_render_variant_cache"):
            self._render_variant_cache = {}

        def render_candidate(reverse_forward: bool, basis_layout: str) -> tuple[torch.Tensor, float, float]:
            c2w_local = build_c2w(
                look_at=center_norm,
                eye=cam_pos,
                reverse_forward=reverse_forward,
                basis_layout=basis_layout,
            )
            test_c2ws_local = torch.from_numpy(c2w_local).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
            rgb_local, _ = gaussian_render(
                gaussian_params,
                test_c2ws_local,
                intr,
                image_width,
                image_height,
                near_plane=near_dynamic,
                far_plane=far_dynamic,
                use_checkpoint=False,
                sh_degree=0,
                bg_mode="black",
            )
            rgb_img_local = rgb_local[0, 0].clamp(-1.0, 1.0).add(1.0).div(2.0)
            # score: prefer non-background coverage + contrast.
            gray = rgb_img_local.mean(dim=0)
            non_bg_ratio = float((gray > 0.03).float().mean().item())
            std_v = float(rgb_img_local.std().item())
            score = non_bg_ratio + 0.5 * std_v
            return rgb_img_local, score, non_bg_ratio

        cached_variant = self._render_variant_cache.get(
            ply_path,
            {"reverse_forward": False, "basis_layout": "row"},
        )
        rgb_img, best_score, best_non_bg_ratio = render_candidate(
            reverse_forward=bool(cached_variant["reverse_forward"]),
            basis_layout=str(cached_variant["basis_layout"]),
        )

        # If the cached/default pose is too empty, probe multiple camera conventions once.
        if best_score < 0.03 or best_non_bg_ratio < 0.001:
            candidates = [
                {"reverse_forward": False, "basis_layout": "row"},
                {"reverse_forward": True, "basis_layout": "row"},
                {"reverse_forward": False, "basis_layout": "col"},
                {"reverse_forward": True, "basis_layout": "col"},
            ]
            best_variant = cached_variant
            for cand in candidates:
                rgb_try, score_try, non_bg_try = render_candidate(
                    reverse_forward=bool(cand["reverse_forward"]),
                    basis_layout=str(cand["basis_layout"]),
                )
                if score_try > best_score:
                    rgb_img = rgb_try
                    best_score = score_try
                    best_non_bg_ratio = non_bg_try
                    best_variant = cand
            self._render_variant_cache[ply_path] = best_variant

        # If gsplat still fails (near-empty), fallback to deterministic point projection.
        if best_score < 0.03 or best_non_bg_ratio < 0.001:
            best_variant = self._render_variant_cache.get(
                ply_path,
                {"reverse_forward": False, "basis_layout": "row"},
            )
            c2w_best = build_c2w(
                look_at=center_norm,
                eye=cam_pos,
                reverse_forward=bool(best_variant["reverse_forward"]),
                basis_layout=str(best_variant["basis_layout"]),
            )

            img_fallback = np.zeros((image_height, image_width, 3), dtype=np.float32)
            depth_buf = np.full((image_height, image_width), np.inf, dtype=np.float32)

            # Optional light subsampling for very dense point clouds.
            max_points = 300000
            if points_norm.shape[0] > max_points:
                rng = np.random.default_rng(42)
                keep_idx = rng.choice(points_norm.shape[0], size=max_points, replace=False)
                proj_points = points_norm[keep_idx]
                proj_colors = colors[keep_idx]
            else:
                proj_points = points_norm
                proj_colors = colors

            w2c = np.linalg.inv(c2w_best).astype(np.float32)
            pts_h = np.concatenate(
                [proj_points, np.ones((proj_points.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            cam_pts = (w2c @ pts_h.T).T[:, :3]

            best_proj_count = -1
            best_proj_payload = None
            for depth_sign in [1.0, -1.0]:
                z = cam_pts[:, 2] * depth_sign
                valid_z = z > 1e-4
                cam_pts_s = cam_pts[valid_z]
                z_s = z[valid_z]
                c_s = proj_colors[valid_z]
                if cam_pts_s.shape[0] == 0:
                    continue

                u = (fx * (cam_pts_s[:, 0] / z_s) + cx).astype(np.int32)
                v = (fy * (cam_pts_s[:, 1] / z_s) + cy).astype(np.int32)
                in_view = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
                view_count = int(in_view.sum())
                if view_count > best_proj_count:
                    best_proj_count = view_count
                    best_proj_payload = (u[in_view], v[in_view], z_s[in_view], c_s[in_view])

            if best_proj_payload is not None and best_proj_count > 0:
                u, v, z, c_proj = best_proj_payload
                order = np.argsort(z)
                u = u[order]
                v = v[order]
                z = z[order]
                c_proj = c_proj[order]

                for uu, vv, zz, cc in zip(u, v, z, c_proj):
                    if zz < depth_buf[vv, uu]:
                        depth_buf[vv, uu] = zz
                        img_fallback[vv, uu] = np.clip(cc, 0.0, 1.0)

                valid_mask = np.isfinite(depth_buf).astype(np.uint8)
                if valid_mask.any():
                    kernel = np.ones((3, 3), np.uint8)
                    dilated = cv2.dilate((img_fallback * 255).astype(np.uint8), kernel, iterations=1)
                    filled = cv2.dilate(valid_mask, kernel, iterations=1)
                    img_fallback[filled > 0] = dilated[filled > 0] / 255.0

                rgb_img = torch.from_numpy(img_fallback).permute(2, 0, 1).to(torch.float32)

        rgb_np = (
            rgb_img.mul(255.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        # Orientation fix: vertical flip only (correct upside-down).
        # "Scene faces camera" is done in 3D by adding 180° to yaw when building the camera.
        rgb_np = np.flipud(rgb_np)
        return Image.fromarray(rgb_np)

    def render_orbit_video_with_3dgs(
        self,
        ply_path: str,
        base_camera_config: Dict[str, Any],
        num_frames: int = 24,
        yaw_step: float = 6.0,
        image_width: int = 704,
        image_height: int = 480,
        fps: int = 12,
        output_path: Optional[str] = None,
    ) -> List[Image.Image]:
        frames: List[Image.Image] = []
        center = base_camera_config.get("center")
        radius = float(base_camera_config.get("radius", 4.0))
        base_yaw = float(base_camera_config.get("yaw", 0.0))
        pitch = float(base_camera_config.get("pitch", 0.0))

        for i in range(num_frames):
            camera_config = {
                "center": center,
                "radius": radius,
                "yaw": base_yaw + i * yaw_step,
                "pitch": pitch,
            }
            frames.append(
                self.render_with_3dgs(
                    ply_path=ply_path,
                    camera_config=camera_config,
                    image_width=image_width,
                    image_height=image_height,
                )
            )

        if output_path is not None and len(frames) > 0:
            self._export_video(frames, output_path, fps=fps)
        return frames

    @staticmethod
    def _apply_interaction_to_camera(
        camera_cfg: Dict[str, Any],
        interaction: str,
        camera_range: Dict[str, Any],
        yaw_step: float = 10.0,
        pitch_step: float = 7.5,
        zoom_factor: float = 0.9,
    ) -> Dict[str, Any]:
        yaw = float(camera_cfg.get("yaw", 0.0))
        pitch = float(camera_cfg.get("pitch", 0.0))
        radius = float(camera_cfg.get("radius", 4.0))

        if interaction in ["move_left", "rotate_left"]:
            yaw -= yaw_step
        elif interaction in ["move_right", "rotate_right"]:
            yaw += yaw_step
        elif interaction == "move_up":
            pitch += pitch_step
        elif interaction == "move_down":
            pitch -= pitch_step
        elif interaction == "zoom_in":
            radius *= zoom_factor
        elif interaction == "zoom_out":
            radius /= zoom_factor

        camera_cfg["yaw"] = max(camera_range["yaw_min"], min(camera_range["yaw_max"], yaw))
        camera_cfg["pitch"] = max(camera_range["pitch_min"], min(camera_range["pitch_max"], pitch))
        camera_cfg["radius"] = max(camera_range["radius_min"], min(camera_range["radius_max"], radius))
        return camera_cfg

    def apply_interaction_to_camera(
        self,
        camera_cfg: Dict[str, Any],
        interaction: str,
        camera_range: Dict[str, Any],
        yaw_step: float = 10.0,
        pitch_step: float = 7.5,
        zoom_factor: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Public wrapper for camera update with interaction signals.
        """
        return self._apply_interaction_to_camera(
            camera_cfg=camera_cfg,
            interaction=interaction,
            camera_range=camera_range,
            yaw_step=yaw_step,
            pitch_step=pitch_step,
            zoom_factor=zoom_factor,
        )

    def render_interaction_video_with_3dgs(
        self,
        ply_path: str,
        camera_range: Dict[str, Any],
        base_camera_config: Dict[str, Any],
        interaction_sequence: List[str],
        image_width: int = 704,
        image_height: int = 480,
        fps: int = 12,
        output_path: Optional[str] = None,
    ) -> List[Image.Image]:
        frames: List[Image.Image] = []
        camera_cfg = {
            "center": base_camera_config.get("center", camera_range["center"]),
            "radius": float(base_camera_config.get("radius", 4.0)),
            "yaw": float(base_camera_config.get("yaw", 0.0)),
            "pitch": float(base_camera_config.get("pitch", 0.0)),
        }

        for sig in interaction_sequence:
            camera_cfg = self._apply_interaction_to_camera(camera_cfg, sig, camera_range)
            frames.append(
                self.render_with_3dgs(
                    ply_path=ply_path,
                    camera_config=camera_cfg,
                    image_width=image_width,
                    image_height=image_height,
                )
            )

        if output_path is not None and len(frames) > 0:
            self._export_video(frames, output_path, fps=fps)
        return frames

    def run_two_stage_3dgs_video(
        self,
        data_path: Union[str, np.ndarray, List[str], List[np.ndarray]],
        interaction: Optional[Union[str, List[str]]] = None,
        output_dir: str = "./vggt_output",
        point_conf_threshold: float = 0.2,
        resolution: int = 518,
        preprocess_mode: str = "crop",
        camera_radius: Optional[float] = None,
        camera_yaw: float = 0.0,
        camera_pitch: float = 0.0,
        camera_view: Optional[List[float]] = None,
        image_width: int = 704,
        image_height: int = 480,
        output_name: str = "vggt_3dgs_demo.mp4",
        fps: int = 12,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        recon_info = self.reconstruct_ply(
            input_=data_path,
            ply_path=output_dir,
            interaction=None,
            point_conf_threshold=point_conf_threshold,
            resolution=resolution,
            preprocess_mode=preprocess_mode,
        )

        ply_path = recon_info["ply_path"]
        camera_range = recon_info["camera_range"]
        default_camera = recon_info["default_camera"]
        base_camera = {
            "center": camera_range["center"],
            "radius": float(camera_radius if camera_radius is not None else default_camera["radius"]),
            "yaw": camera_yaw,
            "pitch": camera_pitch,
        }

        # Apply high-level 5D camera_view if provided.
        base_camera = self._apply_camera_view_to_camera_cfg(
            camera_cfg=base_camera,
            camera_view=camera_view,
            camera_range=camera_range,
        )

        output_video_path = os.path.join(output_dir, output_name)
        interaction_sequence = self._normalize_interaction_sequence(interaction)
        if interaction_sequence:
            self.render_interaction_video_with_3dgs(
                ply_path=ply_path,
                camera_range=camera_range,
                base_camera_config=base_camera,
                interaction_sequence=interaction_sequence,
                image_width=image_width,
                image_height=image_height,
                fps=fps,
                output_path=output_video_path,
            )
        else:
            self.render_orbit_video_with_3dgs(
                ply_path=ply_path,
                base_camera_config=base_camera,
                image_width=image_width,
                image_height=image_height,
                fps=fps,
                output_path=output_video_path,
            )
        return output_video_path

    def run_stage2_3dgs_video_from_reconstruction(
        self,
        recon_info: Dict[str, Any],
        interaction: Optional[Union[str, List[str]]] = None,
        output_dir: str = "./vggt_output",
        camera_radius: Optional[float] = None,
        camera_yaw: float = 0.0,
        camera_pitch: float = 0.0,
        camera_view: Optional[List[float]] = None,
        image_width: int = 704,
        image_height: int = 480,
        output_name: str = "vggt_3dgs_demo.mp4",
        fps: int = 12,
    ) -> str:
        """
        Stage 2 only: render video from existing reconstruction info.
        """
        os.makedirs(output_dir, exist_ok=True)

        ply_path = recon_info["ply_path"]
        camera_range = recon_info["camera_range"]
        default_camera = recon_info["default_camera"]
        base_camera = {
            "center": camera_range["center"],
            "radius": float(camera_radius if camera_radius is not None else default_camera["radius"]),
            "yaw": camera_yaw,
            "pitch": camera_pitch,
        }

        base_camera = self._apply_camera_view_to_camera_cfg(
            camera_cfg=base_camera,
            camera_view=camera_view,
            camera_range=camera_range,
        )

        output_video_path = os.path.join(output_dir, output_name)
        interaction_sequence = self._normalize_interaction_sequence(interaction)
        if interaction_sequence:
            self.render_interaction_video_with_3dgs(
                ply_path=ply_path,
                camera_range=camera_range,
                base_camera_config=base_camera,
                interaction_sequence=interaction_sequence,
                image_width=image_width,
                image_height=image_height,
                fps=fps,
                output_path=output_video_path,
            )
        else:
            self.render_orbit_video_with_3dgs(
                ply_path=ply_path,
                base_camera_config=base_camera,
                image_width=image_width,
                image_height=image_height,
                fps=fps,
                output_path=output_video_path,
            )
        return output_video_path

    def run_two_stage_3dgs_stream_cli(
        self,
        data_path: Union[str, np.ndarray, List[str], List[np.ndarray]],
        output_dir: str = "./vggt_stream_output",
        point_conf_threshold: float = 0.2,
        resolution: int = 518,
        preprocess_mode: str = "crop",
        image_width: int = 704,
        image_height: int = 480,
        fps: int = 12,
        output_name: str = "vggt_stream_demo.mp4",
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        recon_info = self.reconstruct_ply(
            input_=data_path,
            ply_path=output_dir,
            interaction=None,
            point_conf_threshold=point_conf_threshold,
            resolution=resolution,
            preprocess_mode=preprocess_mode,
        )

        available_interactions = [
            "move_left",
            "move_right",
            "move_up",
            "move_down",
            "zoom_in",
            "zoom_out",
            "rotate_left",
            "rotate_right",
        ]

        ply_path = recon_info["ply_path"]
        camera_range = recon_info["camera_range"]
        camera_cfg = dict(recon_info["default_camera"])

        print("Stage-1 reconstruction done.")
        print(f"PLY saved to: {ply_path}")
        print("Camera range:", camera_range)
        print("Default camera:", camera_cfg)
        print("\nAvailable interactions:")
        for i, interaction in enumerate(available_interactions):
            print(f"  {i + 1}. {interaction}")
        print("Tips:")
        print("  - Input multiple interactions separated by comma (e.g., 'move_left,zoom_in')")
        print("  - Input 'n' or 'q' to stop and export video")

        all_frames: List[np.ndarray] = []
        first_frame = self.render_with_3dgs(
            ply_path=ply_path,
            camera_config=camera_cfg,
            image_width=image_width,
            image_height=image_height,
        )
        all_frames.append(np.array(first_frame))

        turn_idx = 0
        print("\n--- VGGT Interactive Stream Started ---")
        while True:
            interaction_input = input(f"\n[Turn {turn_idx}] Enter interaction(s) (or 'n'/'q' to stop): ").strip().lower()
            if interaction_input in ["n", "q"]:
                print("Stopping interaction loop...")
                break

            current_signal = [s.strip() for s in interaction_input.split(",") if s.strip()]
            invalid = [s for s in current_signal if s not in available_interactions]
            if invalid:
                print(f"Invalid interaction(s): {invalid}")
                print(f"Please choose from: {available_interactions}")
                continue
            if not current_signal:
                print("No valid interaction provided. Please try again.")
                continue

            try:
                frames_input = input(f"[Turn {turn_idx}] Enter frame units (e.g., 1 or 2): ").strip()
                frame_units = int(frames_input)
                if frame_units <= 0:
                    print("Frame units must be a positive integer.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
                continue

            for sig in current_signal:
                for _ in range(frame_units * 6):
                    camera_cfg = self._apply_interaction_to_camera(
                        camera_cfg,
                        sig,
                        camera_range,
                        yaw_step=2.0,
                        pitch_step=1.5,
                        zoom_factor=0.98,
                    )
                    frame = self.render_with_3dgs(
                        ply_path=ply_path,
                        camera_config=camera_cfg,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    all_frames.append(np.array(frame))

            print(f"[Turn {turn_idx}] done. Total frames: {len(all_frames)}")
            print(f"Current camera: {camera_cfg}")
            turn_idx += 1

        output_video_path = os.path.join(output_dir, output_name)
        self._export_video(all_frames, output_video_path, fps=fps)
        print(f"Total frames generated: {len(all_frames)}")
        print(f"Stream video saved to: {output_video_path}")
        return output_video_path
    
    def stream(
        self,
        input_: Union[str, np.ndarray, List[str], List[np.ndarray]],
        interaction: Optional[Union[str, Dict[str, Any]]] = None,
        task_type: Optional[str] = None,
        **kwargs,
    ):
        """
        Stream interface.

        task_type:
            - \"vggt_two_stage_3dgs_stream_cli\": run interactive two-stage CLI stream,
              ignoring generator semantics (side-effect only, returns output_video_path).
            - None / other: fallback to one-shot process() and yield image tensors.
        """
        if task_type == "vggt_two_stage_3dgs_stream_cli":
            # Delegate to high-level interactive helper; ignore interaction/kwargs other than config.
            return self.run_two_stage_3dgs_stream_cli(
                data_path=input_,
                **kwargs,
            )

        # Fallback: simple process() and yield images as tensors (backward compatible).
        result = self.process(input_, interaction=interaction, **kwargs)
        for img in result.images:
            yield torch.from_numpy(np.array(img))
    
    def __call__(
        self,
        input_: Union[str, np.ndarray, List[str], List[np.ndarray]],
        interaction: Optional[Union[str, Dict[str, Any], List[str]]] = None,
        task_type: Optional[str] = None,
        **kwargs
    ) -> Union[VGGTResult, str]:
        """
        Main call interface for the pipeline.

        task_type:
            - None or \"vggt_base\": direct VGGT representation/process (returns VGGTResult)
            - \"vggt_two_stage_3dgs\": run two-stage reconstruction + 3DGS video
              (returns output_video_path: str)
        """
        if task_type == "vggt_two_stage_3dgs":
            return self.run_two_stage_3dgs_video(
                data_path=input_,
                interaction=interaction,
                **kwargs,
            )

        # Default: one-stage VGGT representation.
        return self.process(
            input_=input_,
            interaction=interaction,
            **kwargs,
        )


__all__ = ["VGGTPipeline", "VGGTResult"]

