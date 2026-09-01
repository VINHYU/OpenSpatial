#!/usr/bin/env python3
"""Validate, stage, and sequentially upload the four OpenSpatial models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from huggingface_hub import HfApi
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is required. Install it with: pip install -U huggingface_hub"
    ) from exc


@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: Path
    base_model: str
    license: str
    transformers_version: str
    family: str


MODELS = (
    ModelSpec(
        name="OpenSpatial-InternVL2.5-8B",
        source=Path(
            "/pfs/jianhuiliu/project/SpatialReasoning/InternVL/internvl_chat/"
            "work_dirs/spatial_internvl/20260217/"
            "internvl2_5_8b_spatial_debug/checkpoint-5000"
        ),
        base_model="OpenGVLab/InternVL2_5-8B",
        license="mit",
        transformers_version="the version recommended by InternVL2.5",
        family="internvl",
    ),
    ModelSpec(
        name="OpenSpatial-InternVL3-8B",
        source=Path(
            "/pfs/jianhuiliu/project/SpatialReasoning/InternVL/internvl_chat/"
            "work_dirs/spatial_internvl/20260217/"
            "internvl3_8b_spatial_debug/checkpoint-5000"
        ),
        base_model="OpenGVLab/InternVL3-8B",
        license="apache-2.0",
        transformers_version="the version recommended by InternVL3",
        family="internvl",
    ),
    ModelSpec(
        name="OpenSpatial-Qwen2.5-VL-7B",
        source=Path(
            "/pfs/sunhaoze/code/VST_workspace/20260325/"
            "qwen2_5vl_7b_sft_ablation_camera_pose_estimation_relative_direction_llava_2m_20260325/"
            "checkpoints/global_step_5478/hf_ckpt"
        ),
        base_model="Qwen/Qwen2.5-VL-7B-Instruct",
        license="apache-2.0",
        transformers_version=">=4.51.0",
        family="qwen2_5_vl",
    ),
    ModelSpec(
        name="OpenSpatial-Qwen3-VL-8B",
        source=Path(
            "/pfs/jianhuiliu/project/SpatialReasoning/VST_v2/work_dirs/20260225/"
            "qwen3vl_8b_sft/checkpoints/global_step_5460/hf_ckpt"
        ),
        base_model="Qwen/Qwen3-VL-8B-Instruct",
        license="apache-2.0",
        transformers_version=">=4.57.0",
        family="qwen3_vl",
    ),
)

INFERENCE_FILENAMES = {
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "config.json",
    "conversation.py",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}
INFERENCE_PREFIXES = ("configuration_", "modeling_", "tokenization_")
STAGE_MARKER = ".openspatial-upload-stage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload OpenSpatial model checkpoints to Hugging Face in order."
    )
    parser.add_argument(
        "--namespace",
        default=os.getenv("HF_NAMESPACE", "VINHYU"),
        help="Hugging Face user or organization (default: %(default)s)",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".hf-upload",
        help="Persistent staging/cache directory used for resumable uploads",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Workers per model. Models themselves are always uploaded sequentially.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create missing model repositories as private (default: public)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and stage files without network access or repository creation",
    )
    parser.add_argument(
        "--only",
        choices=[model.name for model in MODELS],
        help="Upload only one named model",
    )
    parser.add_argument(
        "--start-at",
        choices=[model.name for model in MODELS],
        help="Skip models before this one; useful when resuming the sequence",
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if args.only and args.start_at:
        parser.error("--only and --start-at cannot be used together")
    return args


def selected_models(args: argparse.Namespace) -> Iterable[ModelSpec]:
    models = list(MODELS)
    if args.only:
        return [model for model in models if model.name == args.only]
    if args.start_at:
        start = next(i for i, model in enumerate(models) if model.name == args.start_at)
        return models[start:]
    return models


def is_inference_file(path: Path) -> bool:
    name = path.name
    return (
        name in INFERENCE_FILENAMES
        or name.startswith(INFERENCE_PREFIXES)
        or (name.startswith("model-") and name.endswith(".safetensors"))
    )


def validate_source(model: ModelSpec) -> list[Path]:
    if not model.source.is_dir():
        raise RuntimeError(f"source directory does not exist: {model.source}")

    files = sorted(
        path for path in model.source.iterdir() if path.is_file() and is_inference_file(path)
    )
    file_by_name = {path.name: path for path in files}
    required = {"config.json", "model.safetensors.index.json", "tokenizer_config.json"}
    missing = sorted(required - file_by_name.keys())
    if missing:
        raise RuntimeError(f"{model.name}: missing required files: {', '.join(missing)}")

    with file_by_name["model.safetensors.index.json"].open() as handle:
        index = json.load(handle)
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        raise RuntimeError(f"{model.name}: weight_map is empty")
    missing_shards = [name for name in shards if name not in file_by_name]
    if missing_shards:
        raise RuntimeError(
            f"{model.name}: index references missing shards: {', '.join(missing_shards)}"
        )

    with file_by_name["config.json"].open() as handle:
        config = json.load(handle)
    if not config.get("architectures"):
        raise RuntimeError(f"{model.name}: config.json has no architectures entry")

    return files


def model_card(model: ModelSpec, repo_id: str) -> str:
    if model.family == "internvl":
        load_example = f'''from transformers import AutoModel, AutoTokenizer

model_id = "{repo_id}"
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(
    model_id, trust_remote_code=True, use_fast=False
)'''
    elif model.family == "qwen2_5_vl":
        load_example = f'''from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

model_id = "{repo_id}"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)'''
    else:
        load_example = f'''from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

model_id = "{repo_id}"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)'''

    return f'''---
license: {model.license}
base_model: {model.base_model}
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- openspatial
- spatial-reasoning
- vision-language
---

# {model.name}

{model.name} is an OpenSpatial model fine-tuned for spatial understanding and
reasoning. This repository contains full, inference-ready model weights.

- Project: https://github.com/VINHYU/OpenSpatial
- Paper: https://arxiv.org/abs/2604.07296
- Training data: https://huggingface.co/datasets/jdopensource/JoyAI-Image-OpenSpatial
- Base model: https://huggingface.co/{model.base_model}

## Loading

Use `transformers` {model.transformers_version} for this checkpoint.

```python
{load_example}
```

Please refer to the OpenSpatial repository and the base model repository for
full usage instructions, limitations, and license terms.
'''


def prepare_stage(model: ModelSpec, files: list[Path], stage_root: Path, repo_id: str) -> Path:
    stage = stage_root / repo_id
    stage.mkdir(parents=True, exist_ok=True)
    marker = stage / STAGE_MARKER
    if any(stage.iterdir()) and not marker.exists():
        raise RuntimeError(
            f"refusing to use non-empty unmarked staging directory: {stage}"
        )
    marker.touch(exist_ok=True)

    wanted = {path.name for path in files}
    for path in stage.iterdir():
        if path.is_symlink() and path.name not in wanted:
            path.unlink()

    for source in files:
        destination = stage / source.name
        if destination.is_symlink():
            if destination.resolve() == source.resolve():
                continue
            destination.unlink()
        elif destination.exists():
            raise RuntimeError(f"staging path already exists and is not a symlink: {destination}")
        destination.symlink_to(source.resolve())

    (stage / "README.md").write_text(model_card(model, repo_id), encoding="utf-8")
    return stage


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def main() -> int:
    args = parse_args()
    models = list(selected_models(args))
    prepared: list[tuple[ModelSpec, str, Path, list[Path]]] = []

    print("Validating and staging model files...")
    for model in models:
        files = validate_source(model)
        repo_id = f"{args.namespace}/{model.name}"
        stage = prepare_stage(model, files, args.stage_dir.resolve(), repo_id)
        total = sum(path.stat().st_size for path in files)
        print(f"  {repo_id}: {len(files)} files, {human_size(total)} -> {stage}")
        prepared.append((model, repo_id, stage, files))

    if args.dry_run:
        print("Dry run complete; no Hugging Face repositories were created or modified.")
        return 0

    api = HfApi()
    identity = api.whoami()
    print(f"Authenticated on Hugging Face as {identity.get('name', '<unknown>')}.")

    for position, (model, repo_id, stage, _) in enumerate(prepared, start=1):
        print(f"\n[{position}/{len(prepared)}] Uploading {model.name} to {repo_id}")
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=stage,
            private=args.private,
            ignore_patterns=[STAGE_MARKER, ".cache/**"],
            num_workers=args.num_workers,
        )
        print(f"Completed: https://huggingface.co/{repo_id}")

    print("\nAll requested models were uploaded successfully.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
