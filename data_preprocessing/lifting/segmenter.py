"""SAM 3 box-prompt instance segmentation."""

from contextlib import nullcontext
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import torch

from .config import LiftingConfig


class SAM3InstanceSegmenter:
    """Load SAM 3 and turn detector boxes into binary masks."""

    def __init__(self, config: LiftingConfig):
        self.config = config
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise ImportError(
                "Install the lifting dependencies from "
                "data_preprocessing/lifting/requirements.txt"
            ) from exc

        print(f"[SAM3] Loading on {config.device}")
        self.model = build_sam3_image_model(
            bpe_path=config.sam3_vocab_path,
            device=config.device,
            enable_inst_interactivity=True,
            checkpoint_path=config.sam3_checkpoint,
            load_from_HF=config.sam3_checkpoint is None,
        )
        self.model = self.model.to(config.device)
        self.processor = Sam3Processor(self.model)

    def segment(
        self, image: Image.Image, detections: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        if not detections:
            return []

        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.config.device.startswith("cuda")
            else nullcontext()
        )
        with context:
            state = self.processor.set_image(image)
            boxes = np.asarray([item["box_2d"] for item in detections])
            masks, _, _ = self.model.predict_inst(
                state,
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
            )

        result = []
        for mask in masks:
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            result.append(mask.astype(bool))
        return result
