"""Runtime configuration for the 3D lifting pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LiftingConfig:
    output_base_dir: str
    api_key: str
    sam3_checkpoint: Optional[str] = None
    sam3_vocab_path: Optional[str] = None
    da3_model: str = "depth-anything/DA3NESTED-GIANT-LARGE"
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    detector_model: str = "gemini-3-pro-preview"
    api_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    device: str = "cuda"
    process_resolution: int = 504
    target_frame_num: int = 100
    num_detection_views: int = 80
    detection_workers: int = 20
    use_detection_cache: bool = False
    api_timeout: int = 300
    api_max_retries: int = 50
    spatial_iou_threshold: float = 0.05
    cross_category_iou_threshold: float = 0.2
    semantic_similarity_threshold: float = 0.4
    skip_merge_labels: tuple = ("floor", "ground")
    voxel_size: float = 0.05
    object_downsample_voxel_size: float = 0.0
    scene_downsample_voxel_size: float = 0.0
    detection_prompt: str = (
        "Detect the objects visible in the image. Never return masks or code fencing. "
        "Output a JSON list of objects where each entry contains the 2D bounding box "
        "in the key 'box_2d' (format: [ymin, xmin, ymax, xmax], normalized 0-1000), "
        "the text label in the key 'label' (broad category name in singular form, "
        "e.g. 'chair', 'table', 'floor'), and a detailed visual description in the "
        "key 'description'."
    )
