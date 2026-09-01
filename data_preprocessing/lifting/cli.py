"""Command-line interface for OpenSpatial 3D lifting."""

import argparse
import glob
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Tuple

import torch

from .config import LiftingConfig


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v")


def _sample_evenly(paths: List[str], maximum: Optional[int]) -> List[str]:
    if maximum is None or len(paths) <= maximum:
        return paths
    if maximum <= 0:
        raise ValueError("--max-frames must be positive")
    step = len(paths) / maximum
    return [paths[int(index * step)] for index in range(maximum)]


def _extract_video(
    video_path: str,
    fps: Optional[float],
    max_frames: Optional[int],
) -> Tuple[List[str], str]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for video input")
    temp_dir = tempfile.mkdtemp(prefix="openspatial_lifting_")
    command = ["ffmpeg", "-i", video_path]
    if fps is not None:
        command.extend(["-vf", f"fps={fps}"])
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.extend([
        "-q:v",
        "2",
        os.path.join(temp_dir, "frame_%06d.jpg"),
        "-y",
    ])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        shutil.rmtree(temp_dir)
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    return sorted(glob.glob(os.path.join(temp_dir, "frame_*.jpg"))), temp_dir


def load_input(
    input_path: str,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> Tuple[str, List[str], Optional[str]]:
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    if os.path.isfile(input_path):
        suffix = os.path.splitext(input_path)[1].lower()
        scene_name = os.path.splitext(os.path.basename(input_path))[0]
        if suffix in IMAGE_EXTENSIONS:
            return scene_name, [input_path], None
        if suffix in VIDEO_EXTENSIONS:
            images, temp_dir = _extract_video(input_path, fps, max_frames)
            return scene_name, images, temp_dir
        raise ValueError(f"Unsupported input file type: {suffix}")

    paths = sorted(
        path
        for path in glob.glob(os.path.join(input_path, "*"))
        if os.path.isfile(path)
        and os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
    )
    return (
        os.path.basename(os.path.normpath(input_path)),
        _sample_evenly(paths, max_frames),
        None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lift an image sequence or video into a metric 3D scene."
    )
    parser.add_argument("--input", required=True, help="Image, image directory, or video")
    parser.add_argument("--output", required=True, help="Output root directory")
    parser.add_argument("--sam3-checkpoint", default=None, help="Path to sam3.pt")
    parser.add_argument(
        "--sam3-vocab",
        default=None,
        help="Optional SAM 3 BPE vocabulary path; packaged vocabulary is used by default",
    )
    parser.add_argument(
        "--da3-model",
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="DA3 Hugging Face model ID or local checkpoint directory",
    )
    parser.add_argument(
        "--semantic-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model ID or local directory",
    )
    parser.add_argument(
        "--detector-model",
        default="gemini-3-pro-preview",
        help="Gemini model used for 2D object detection",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY"),
        help="Gemini API key; defaults to GEMINI_API_KEY",
    )
    parser.add_argument(
        "--api-base-url",
        default="https://generativelanguage.googleapis.com/v1beta",
        help="Gemini generateContent API base URL",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda, cuda:1, or cpu",
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--target-frames", type=int, default=100)
    parser.add_argument("--detection-views", type=int, default=80)
    parser.add_argument("--process-resolution", type=int, default=504)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--name", default=None, help="Optional scene name")
    parser.add_argument(
        "--cache-detections",
        action="store_true",
        help="Reuse cached detector responses for the same absolute input paths",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep temporary frames extracted from video",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.api_key:
        parser.error("--api-key or GEMINI_API_KEY is required")
    if args.sam3_checkpoint and not os.path.isfile(args.sam3_checkpoint):
        parser.error(f"SAM 3 checkpoint not found: {args.sam3_checkpoint}")
    if args.target_frames <= 0 or args.detection_views <= 0 or args.workers <= 0:
        parser.error("--target-frames, --detection-views, and --workers must be positive")

    scene_name, image_paths, temporary_frames = load_input(
        args.input, args.fps, args.max_frames
    )
    if args.name:
        scene_name = args.name
    if not image_paths:
        parser.error(f"No supported images found in {args.input}")

    config = LiftingConfig(
        output_base_dir=os.path.abspath(args.output),
        api_key=args.api_key,
        sam3_checkpoint=args.sam3_checkpoint,
        sam3_vocab_path=args.sam3_vocab,
        da3_model=args.da3_model,
        semantic_model=args.semantic_model,
        detector_model=args.detector_model,
        api_base_url=args.api_base_url,
        device=args.device,
        process_resolution=args.process_resolution,
        target_frame_num=args.target_frames,
        num_detection_views=args.detection_views,
        detection_workers=args.workers,
        use_detection_cache=args.cache_detections,
    )

    try:
        from .pipeline import SceneLiftingPipeline

        output_dir = SceneLiftingPipeline(config).process_scene(
            scene_name, image_paths
        )
        print(output_dir)
        return 0
    finally:
        if temporary_frames and not args.keep_frames:
            shutil.rmtree(temporary_frames, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
