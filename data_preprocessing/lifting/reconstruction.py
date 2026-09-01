"""Metric multi-view reconstruction with Depth Anything 3."""

import os
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch

from .config import LiftingConfig


class SceneReconstructor:
    """Run DA3 and back-project its depth into a structured world point cloud."""

    def __init__(self, config: LiftingConfig):
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise ImportError(
                "Install the lifting dependencies from "
                "data_preprocessing/lifting/requirements.txt"
            ) from exc

        print(f"[DA3] Loading {config.da3_model} on {self.device}")
        self.model = DepthAnything3.from_pretrained(config.da3_model)
        self.model = self.model.to(self.device)
        self.model.eval()

    def reconstruct(
        self,
        image_paths: List[str],
        save_dir: str,
        return_prediction: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[object]]:
        if not image_paths:
            empty = np.zeros((0, 0, 0, 3))
            return empty, empty, None

        print(f"[DA3] Running inference on {len(image_paths)} images")
        prediction = self.model.inference(
            image=image_paths,
            process_res=self.config.process_resolution,
            process_res_method="upper_bound_resize",
            export_dir=None,
            export_format="glb",
        )

        num_views = len(image_paths)
        height, width = prediction.depth[0].shape
        points_structure = np.full(
            (num_views, height, width, 3), np.nan, dtype=np.float32
        )
        colors_structure = np.zeros(
            (num_views, height, width, 3), dtype=np.uint8
        )
        all_valid_points = []
        all_valid_colors = []

        for index in range(num_views):
            depth = prediction.depth[index]
            confidence = prediction.conf[index]
            intrinsic = prediction.intrinsics[index]
            world_to_camera = prediction.extrinsics[index]
            rgb = prediction.processed_images[index]

            height, width = depth.shape
            confidence_threshold = np.percentile(confidence, 15)
            valid = (confidence > confidence_threshold) & (depth > 0)
            v_grid, u_grid = np.meshgrid(
                np.arange(height), np.arange(width), indexing="ij"
            )
            pixels = np.stack(
                [u_grid.ravel(), v_grid.ravel(), np.ones(height * width)], axis=0
            )
            camera_points = (
                np.linalg.inv(intrinsic) @ pixels
            ) * depth.ravel()
            w2c_homogeneous = np.vstack([world_to_camera, [0, 0, 0, 1]])
            camera_to_world = np.linalg.inv(w2c_homogeneous)
            camera_points_h = np.vstack(
                [camera_points, np.ones((1, camera_points.shape[1]))]
            )
            world_points = (camera_to_world @ camera_points_h)[:3].T
            world_grid = world_points.reshape(height, width, 3)
            world_grid[~valid] = np.nan

            points_structure[index] = world_grid
            colors_structure[index] = rgb

            downsample = 2
            valid_downsampled = valid[::downsample, ::downsample]
            all_valid_points.append(
                world_grid[::downsample, ::downsample][valid_downsampled]
            )
            all_valid_colors.append(
                rgb[::downsample, ::downsample][valid_downsampled]
            )

        if all_valid_points:
            points = np.vstack(all_valid_points)
            colors = np.vstack(all_valid_colors)
            self._save_ply(points, colors, os.path.join(save_dir, "scene_stitched.ply"))

        if return_prediction:
            return points_structure, colors_structure, prediction
        return points_structure, colors_structure, None

    @staticmethod
    def _save_ply(points: np.ndarray, colors: np.ndarray, path: str) -> None:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(
            colors / 255.0 if colors.max() > 1.0 else colors
        )
        o3d.io.write_point_cloud(path, point_cloud)
