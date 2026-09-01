"""Output serialization for 3D lifting and OpenSpatial interoperability."""

from datetime import datetime
import json
import os
import shutil
from typing import Dict, List, Optional

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from PIL import Image


class OutputManager:
    """Write original lifting artifacts plus an OpenSpatial-compatible Parquet."""

    def __init__(self, base_dir: str, scene_name: str, timestamp: Optional[str] = None):
        safe_name = self._sanitize_name(scene_name)
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{safe_name}_{timestamp}"
        self.output_dir = os.path.abspath(os.path.join(base_dir, self.run_name))
        self.images_dir = os.path.join(self.output_dir, "images")
        self.processed_images_dir = os.path.join(self.output_dir, "processed_images")
        self.detections_dir = os.path.join(self.output_dir, "detections")
        self.pointclouds_dir = os.path.join(self.output_dir, "pointclouds")
        self.view_pointclouds_dir = os.path.join(self.output_dir, "view_pointclouds")
        self.masks_dir = os.path.join(self.output_dir, "masks")
        self.depths_dir = os.path.join(self.output_dir, "depths")
        self.confs_dir = os.path.join(self.output_dir, "confs")
        self.cameras_dir = os.path.join(self.output_dir, "cameras")
        for directory in (
            self.output_dir,
            self.images_dir,
            self.processed_images_dir,
            self.detections_dir,
            self.pointclouds_dir,
            self.view_pointclouds_dir,
            self.masks_dir,
            self.depths_dir,
            self.confs_dir,
            self.cameras_dir,
        ):
            os.makedirs(directory, exist_ok=True)

        self.metadata = {
            "scene_name": scene_name,
            "run_name": self.run_name,
            "created_at": datetime.now().isoformat(),
            "num_input_frames": 0,
            "num_selected_views": 0,
            "num_detection_views": 0,
            "num_objects": 0,
        }
        self.camera_extrinsics = {}
        self.camera_intrinsics = {}

    @staticmethod
    def _sanitize_name(name: str) -> str:
        for char in '<>:"/\\|?*':
            name = name.replace(char, "_")
        return name.strip()[:100] or "scene"

    def save_frame(
        self,
        frame_index: int,
        source_path: str,
        processed_image: np.ndarray,
        depth: np.ndarray,
        confidence: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
    ) -> Dict[str, str]:
        source_suffix = os.path.splitext(source_path)[1].lower() or ".jpg"
        source_output = os.path.join(
            self.images_dir, f"frame_{frame_index:03d}{source_suffix}"
        )
        shutil.copy2(source_path, source_output)

        processed_output = os.path.join(
            self.processed_images_dir, f"frame_{frame_index:03d}.png"
        )
        Image.fromarray(np.asarray(processed_image, dtype=np.uint8)).save(processed_output)

        depth_output = os.path.join(self.depths_dir, f"frame_{frame_index:03d}.png")
        depth_mm = np.clip(depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        if not cv2.imwrite(depth_output, depth_mm):
            raise OSError(f"Failed to save depth map: {depth_output}")

        confidence_output = os.path.join(
            self.confs_dir, f"frame_{frame_index:03d}.npy"
        )
        np.save(confidence_output, confidence)

        self.camera_extrinsics[frame_index] = np.asarray(extrinsics)
        self.camera_intrinsics[frame_index] = np.asarray(intrinsics)
        return {
            "source_image": source_output,
            "image": processed_output,
            "depth_map": depth_output,
            "confidence": confidence_output,
        }

    def save_detection_result(
        self,
        frame_index: int,
        detections: List[Dict],
        image_path: str,
        width: int,
        height: int,
    ) -> None:
        output = {
            "frame_index": frame_index,
            "original_image_path": image_path,
            "image_width": width,
            "image_height": height,
            "num_detections": len(detections),
            "detections": detections,
        }
        path = os.path.join(self.detections_dir, f"frame_{frame_index:03d}.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(output, file, indent=2)

    def save_all_detections_summary(self, detections_by_frame: Dict[int, Dict]) -> None:
        labels = set()
        for data in detections_by_frame.values():
            labels.update(
                item.get("label", "unknown").lower().strip().replace(" ", "_")
                for item in data.get("detections", [])
            )
        payload = {
            "total_frames_processed": len(detections_by_frame),
            "unique_object_categories": len(labels),
            "categories": sorted(labels),
        }
        with open(
            os.path.join(self.output_dir, "detections_summary.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(payload, file, indent=2)

    @staticmethod
    def _homogeneous_w2c(world_to_camera: np.ndarray) -> np.ndarray:
        world_to_camera = np.asarray(world_to_camera)
        if world_to_camera.shape == (4, 4):
            return world_to_camera.copy()
        if world_to_camera.shape != (3, 4):
            raise ValueError(
                f"Expected a 3x4 or 4x4 extrinsic matrix, got {world_to_camera.shape}"
            )
        return np.vstack([world_to_camera, [0, 0, 0, 1]])

    @classmethod
    def _transform_w2c_for_scene(
        cls, world_to_camera: np.ndarray, transform: Dict
    ) -> np.ndarray:
        w2c = cls._homogeneous_w2c(world_to_camera)
        rotation = np.asarray(transform["rotation_matrix"])
        total_rotation = rotation if transform.get("has_rotation", True) else np.eye(3)
        if transform.get("flip_z", False):
            total_rotation = np.asarray(transform["flip_matrix"]) @ total_rotation

        total_rotation_inv = np.linalg.inv(total_rotation)
        shift = np.array([0.0, 0.0, transform.get("z_shift", 0.0)])
        transformed = np.eye(4)
        transformed[:3, :3] = w2c[:3, :3] @ total_rotation_inv
        transformed[:3, 3] = (
            w2c[:3, 3] - w2c[:3, :3] @ total_rotation_inv @ shift
        )
        return transformed

    @staticmethod
    def _homogeneous_intrinsic(intrinsic: np.ndarray) -> np.ndarray:
        intrinsic = np.asarray(intrinsic)
        if intrinsic.shape == (4, 4):
            return intrinsic.copy()
        if intrinsic.shape != (3, 3):
            raise ValueError(
                f"Expected a 3x3 or 4x4 intrinsic matrix, got {intrinsic.shape}"
            )
        result = np.eye(4)
        result[:3, :3] = intrinsic
        return result

    def finalize_camera_data(self, scene_transform: Dict) -> Dict[int, Dict[str, str]]:
        camera_json = {
            "num_frames": len(self.camera_extrinsics),
            "coordinate_system": "OpenCV camera coordinates; transformed Z-up world",
            "extrinsics_convention": "W2C",
            "frames": {},
        }
        paths = {}
        for index in sorted(self.camera_extrinsics):
            w2c = self._transform_w2c_for_scene(
                self.camera_extrinsics[index], scene_transform
            )
            c2w = np.linalg.inv(w2c)
            intrinsic = self._homogeneous_intrinsic(self.camera_intrinsics[index])
            pose_path = os.path.join(self.cameras_dir, f"frame_{index:03d}_pose.txt")
            intrinsic_path = os.path.join(
                self.cameras_dir, f"frame_{index:03d}_intrinsic.txt"
            )
            np.savetxt(pose_path, c2w)
            np.savetxt(intrinsic_path, intrinsic)
            paths[index] = {"pose": pose_path, "intrinsic": intrinsic_path}
            camera_json["frames"][f"frame_{index:03d}"] = {
                "frame_index": index,
                "extrinsics": {"W2C": w2c[:3].tolist()},
                "intrinsics": {"K": self.camera_intrinsics[index].tolist()},
            }

        with open(
            os.path.join(self.output_dir, "camera_extrinsics.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(camera_json, file, indent=2)
        return paths

    def _save_view_object(
        self,
        frame_index: int,
        object_index: int,
        mask: np.ndarray,
        points_world: np.ndarray,
        colors: np.ndarray,
        world_to_camera: np.ndarray,
    ) -> Dict[str, object]:
        mask_path = os.path.join(
            self.masks_dir,
            f"frame_{frame_index:03d}_object_{object_index:03d}.png",
        )
        binary_mask = np.where(mask, 255, 0).astype(np.uint8)
        Image.fromarray(binary_mask, mode="L").save(mask_path)

        ys, xs = np.where(mask)
        bbox = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ]

        valid = ~np.isnan(points_world).any(axis=1)
        points_world = points_world[valid]
        colors = colors[valid]
        w2c = self._homogeneous_w2c(world_to_camera)
        points_h = np.hstack([points_world, np.ones((len(points_world), 1))])
        points_camera = (w2c @ points_h.T).T[:, :3]

        pointcloud_path = os.path.join(
            self.view_pointclouds_dir,
            f"frame_{frame_index:03d}_object_{object_index:03d}.pcd",
        )
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_camera)
        if len(colors):
            point_cloud.colors = o3d.utility.Vector3dVector(
                colors / 255.0 if colors.max() > 1.0 else colors
            )
        o3d.io.write_point_cloud(pointcloud_path, point_cloud)
        return {
            "mask": mask_path,
            "bbox_2d": bbox,
            "pointcloud": pointcloud_path,
        }

    def save_openspatial_parquet(
        self,
        view_records: List[Dict],
        aggregator,
        camera_paths: Dict[int, Dict[str, str]],
    ) -> str:
        images = []
        source_images = []
        depths = []
        poses = []
        intrinsics = []
        tags_by_view = []
        masks_by_view = []
        boxes_2d_by_view = []
        boxes_3d_by_view = []
        pointclouds_by_view = []
        descriptions_by_view = []
        instance_ids_by_view = []

        for record in view_records:
            index = record["frame_index"]
            images.append(record["paths"]["image"])
            source_images.append(record["paths"]["source_image"])
            depths.append(record["paths"]["depth_map"])
            poses.append(camera_paths[index]["pose"])
            intrinsics.append(camera_paths[index]["intrinsic"])

            view_tags = []
            view_masks = []
            view_boxes_2d = []
            view_boxes_3d = []
            view_pointclouds = []
            view_descriptions = []
            view_instance_ids = []
            seen_ids = set()

            for object_index, (detection, mask, instance_ref) in enumerate(zip(
                record["detections"], record["masks"], record["instance_refs"]
            )):
                instance = aggregator.resolve_instance(instance_ref)
                if instance is None or instance.id in seen_ids:
                    continue
                box = aggregator.bounding_boxes_by_id.get(instance.id)
                if box is None:
                    continue

                mask = np.asarray(mask).astype(bool)
                artifacts = self._save_view_object(
                    index,
                    object_index,
                    mask,
                    record["points"][mask],
                    record["colors"][mask],
                    record["extrinsics"],
                )
                seen_ids.add(instance.id)
                view_tags.append(box["label"])
                view_masks.append(artifacts["mask"])
                view_boxes_2d.append(artifacts["bbox_2d"])
                view_boxes_3d.append(
                    box["center"] + box["size"] + [box["rotation"], 0.0, 0.0]
                )
                view_pointclouds.append(artifacts["pointcloud"])
                view_descriptions.append(box.get("description", ""))
                view_instance_ids.append(instance.id)

            tags_by_view.append(view_tags)
            masks_by_view.append(view_masks)
            boxes_2d_by_view.append(view_boxes_2d)
            boxes_3d_by_view.append(view_boxes_3d)
            pointclouds_by_view.append(view_pointclouds)
            descriptions_by_view.append(view_descriptions)
            instance_ids_by_view.append(view_instance_ids)

        row = {
            "image": images,
            "source_image": source_images,
            "depth_map": depths,
            "pose": poses,
            "intrinsic": intrinsics,
            "depth_scale": [1000] * len(images),
            "is_metric_depth": True,
            "obj_tags": tags_by_view,
            "masks": masks_by_view,
            "bboxes_2d": boxes_2d_by_view,
            "bboxes_3d_world_coords": boxes_3d_by_view,
            "pointclouds": pointclouds_by_view,
            "descriptions": descriptions_by_view,
            "instance_ids": instance_ids_by_view,
        }
        parquet_path = os.path.join(self.output_dir, "openspatial.parquet")
        pd.DataFrame([row]).to_parquet(parquet_path, engine="pyarrow")
        return parquet_path

    def finalize(self, object_count: int, parquet_path: str) -> None:
        self.metadata["num_objects"] = object_count
        self.metadata["completed_at"] = datetime.now().isoformat()
        with open(
            os.path.join(self.output_dir, "scene_metadata.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(self.metadata, file, indent=2)

        summary = {
            "run_name": self.run_name,
            "output_directory": self.output_dir,
            "openspatial_parquet": parquet_path,
            "num_objects": object_count,
            "contents": {
                "images": len(os.listdir(self.images_dir)),
                "detection_files": len(os.listdir(self.detections_dir)),
                "pointclouds": len([
                    name for name in os.listdir(self.pointclouds_dir)
                    if name.endswith(".ply")
                ]),
            },
        }
        with open(
            os.path.join(self.output_dir, "summary.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(summary, file, indent=2)
