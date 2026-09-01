"""End-to-end OpenSpatial 3D lifting pipeline."""

from datetime import datetime
from typing import List

import numpy as np
from PIL import Image

from .aggregator import PointCloudAggregator
from .config import LiftingConfig
from .detector import GeminiObjectDetector
from .output_manager import OutputManager
from .reconstruction import SceneReconstructor
from .segmenter import SAM3InstanceSegmenter


class SceneLiftingPipeline:
    """Reconstruct, detect, segment, aggregate, and export one scene."""

    def __init__(self, config: LiftingConfig):
        self.config = config
        self.reconstructor = SceneReconstructor(config)
        self.detector = GeminiObjectDetector(config)
        self.segmenter = SAM3InstanceSegmenter(config)

    @staticmethod
    def _resize_masks(masks, height: int, width: int):
        return [
            np.asarray(
                Image.fromarray(mask).resize((width, height), resample=Image.NEAREST)
            ).astype(bool)
            for mask in masks
        ]

    def process_scene(self, scene_name: str, image_paths: List[str]) -> str:
        if not image_paths:
            raise ValueError("No input images were found")

        manager = OutputManager(
            self.config.output_base_dir,
            scene_name,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        manager.metadata["num_input_frames"] = len(image_paths)

        target_count = min(self.config.target_frame_num, len(image_paths))
        selected = np.unique(
            np.linspace(0, len(image_paths) - 1, target_count, dtype=int)
        )
        reconstruction_images = [image_paths[index] for index in selected]
        manager.metadata["num_selected_views"] = len(reconstruction_images)

        points, colors, prediction = self.reconstructor.reconstruct(
            reconstruction_images,
            manager.pointclouds_dir,
            return_prediction=True,
        )

        detection_count = min(
            self.config.num_detection_views, len(reconstruction_images)
        )
        detection_indices = np.unique(
            np.linspace(
                0, len(reconstruction_images) - 1, detection_count, dtype=int
            )
        )
        manager.metadata["num_detection_views"] = len(detection_indices)

        tasks = []
        frame_paths = {}
        for frame_index, reconstruction_index in enumerate(detection_indices):
            source_path = reconstruction_images[reconstruction_index]
            source_image = Image.open(source_path).convert("RGB")
            tasks.append(
                (
                    frame_index,
                    source_path,
                    source_image.width,
                    source_image.height,
                )
            )
            frame_paths[frame_index] = manager.save_frame(
                frame_index,
                source_path,
                prediction.processed_images[reconstruction_index],
                prediction.depth[reconstruction_index],
                prediction.conf[reconstruction_index],
                prediction.extrinsics[reconstruction_index],
                prediction.intrinsics[reconstruction_index],
            )

        detections_by_index = self.detector.detect_batch(
            tasks, use_cache=self.config.use_detection_cache
        )
        detection_summary = {}
        aggregator = PointCloudAggregator(self.config)
        view_records = []

        for frame_index, reconstruction_index in enumerate(detection_indices):
            source_path, detections = detections_by_index[frame_index]
            source_image = Image.open(source_path).convert("RGB")
            manager.save_detection_result(
                frame_index,
                detections,
                source_path,
                source_image.width,
                source_image.height,
            )
            detection_summary[frame_index] = {
                "image_path": source_path,
                "detections": detections,
            }

            masks = self.segmenter.segment(source_image, detections)
            height, width = points[reconstruction_index].shape[:2]
            masks = self._resize_masks(masks, height, width)
            instance_refs = aggregator.ingest_frame(
                points[reconstruction_index],
                colors[reconstruction_index],
                detections,
                masks,
            )
            view_records.append({
                "frame_index": frame_index,
                "paths": frame_paths[frame_index],
                "detections": detections,
                "masks": masks,
                "instance_refs": instance_refs,
                "points": points[reconstruction_index],
                "colors": colors[reconstruction_index],
                "extrinsics": prediction.extrinsics[reconstruction_index],
            })

        manager.save_all_detections_summary(detection_summary)
        scene_points = points.reshape(-1, 3)
        scene_colors = colors.reshape(-1, 3)
        valid = ~np.isnan(scene_points).any(axis=1)
        scene_points = scene_points[valid]
        scene_colors = scene_colors[valid]

        aggregator.consolidate_instances()
        scene_transform = aggregator.save_results(
            manager.pointclouds_dir, scene_points, scene_colors
        )
        camera_paths = manager.finalize_camera_data(scene_transform)
        parquet_path = manager.save_openspatial_parquet(
            view_records, aggregator, camera_paths
        )
        manager.finalize(len(aggregator.instances), parquet_path)
        print(f"[Lifting] Output saved to {manager.output_dir}")
        return manager.output_dir
