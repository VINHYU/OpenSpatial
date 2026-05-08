# EmbodiedScan Data Preprocessing

Preprocessing pipeline for extracting multi-view 3D perception data from [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan) datasets and converting them into the OpenSpatial Parquet format.

Supports 4 datasets: **ScanNet**, **3RScan**, **Matterport3D**, **ARKitScenes**.

## Installation

```bash
# Install EmbodiedScan from source (required dependency).
# The [visual] extras pull in open3d, which embodiedscan.explorer imports at module load.
git clone https://github.com/OpenRobotLab/EmbodiedScan.git
cd EmbodiedScan
# Workaround for upstream packaging bug: top-level `embodiedscan/` lacks an
# __init__.py so modern setuptools' find_packages() registers nothing on
# `pip install -e`. Tracked in InternRobotics/EmbodiedScan#117 — drop this
# line once it's merged.
touch embodiedscan/__init__.py
pip install -e ".[visual]"
cd ..

# Install this preprocessing package
cd OpenSpatial/data_preprocessing/embodiedscan
pip install -e .
```

## Prerequisites

1. **Raw dataset files** -- Download the original datasets following the [EmbodiedScan data documentation](https://github.com/OpenRobotLab/EmbodiedScan/blob/main/data/README.md). You only need to download the datasets you plan to use.

2. **EmbodiedScan annotation files** -- Download the `.pkl` annotation files and place them as shown below.

## Data Directory Structure

The `--data-root` argument should point to a directory with the following layout:

```
<data-root>/
├── scannet/
│   ├── posed_images/
│   │   └── <scene_id>/              # e.g., scene0000_01
│   │       ├── 00000.jpg             # RGB image
│   │       ├── 00000.png             # depth map (16-bit)
│   │       └── ...
│   └── scans/
│       └── <scene_id>/
│           └── intrinsic/
│               └── intrinsic_depth.txt   # 4x4 intrinsic matrix
│
├── 3rscan/
│   └── <scene_uuid>/
│       └── sequence/
│           ├── _info.txt             # contains m_calibrationDepthIntrinsic
│           ├── frame-000000.color.jpg
│           └── ...
│
├── matterport3d/
│   └── <building_id>/
│       ├── region_segmentations/
│       │   └── *.ply
│       ├── matterport_color_images/
│       │   └── *.jpg
│       ├── matterport_camera_intrinsics/
│       │   └── *.txt                 # width height fx fy cx cy ...
│       └── matterport_depth_images/
│           └── *.png                 # 16-bit depth (depth_scale=4000)
│
├── arkitscenes/
│   └── <scene_id>/
│       └── <scene_id>_frames/
│           ├── lowres_wide/
│           │   └── *.jpg
│           ├── lowres_wide_intrinsics/
│           │   └── *.pincam          # width height fx fy cx cy
│           └── lowres_depth/
│               └── *.png
│
├── embodiedscan_infos_train.pkl      # EmbodiedScan v1 annotations
├── embodiedscan_infos_val.pkl
└── embodiedscan_infos_test.pkl
```

ARKitScenes requires v2 annotations in a sibling directory:

```
<data-root>/../embodiedscan-v2/
├── embodiedscan_infos_train.pkl
├── embodiedscan_infos_val.pkl
└── embodiedscan_infos_test.pkl
```

## Usage

The pipeline has 4 steps: **extract** -> **merge** -> **export** -> **validate**.

### Step 1: Extract (per-image)

```bash
# Single dataset
python -m embodiedscan_data extract \
  --dataset scannet \
  --data-root /path/to/data \
  --output ./output \
  --workers 24

# All datasets
python -m embodiedscan_data extract \
  --dataset all \
  --data-root /path/to/data \
  --output ./output \
  --workers 24

# Smoke test (limit scenes)
python -m embodiedscan_data extract \
  --dataset scannet \
  --data-root /path/to/data \
  --output ./output \
  --workers 4 \
  --max-scenes 2
```

Outputs per-image JSONL files (e.g., `scannet.jsonl`). Supports resume -- rerunning skips already-extracted records.

### Step 2: Merge (per-scene)

```bash
python -m embodiedscan_data merge --input ./output
```

Groups per-image records by `scene_id` into per-scene JSONL files (e.g., `scannet_scenes.jsonl`).

### Step 3: Export (to Parquet)

```bash
python -m embodiedscan_data export --input ./output --format both
```

Converts JSONL to sharded Parquet files under `per_image/` and `per_scene/` subdirectories.

### Step 4: Validate

```bash
python -m embodiedscan_data validate \
  --input ./output \
  --data-root /path/to/data
```

Checks schema completeness, record counts, value ranges, and file path reachability.

## Output Schema

### Per-image record

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique record ID (e.g., `scannet__scene0000_01__00000`) |
| `dataset` | `str` | Source dataset name |
| `scene_id` | `str` | Scene identifier |
| `image` | `str` | RGB image path (relative to `--data-root`) |
| `depth_map` | `str` | Depth map path |
| `pose` | `str` | 4x4 camera-to-world extrinsic matrix (txt) |
| `intrinsic` | `str` | 4x4 camera intrinsic matrix (txt) |
| `depth_scale` | `int` | Depth scale factor (1000 or 4000) |
| `bboxes_3d_world_coords` | `list[list[float]]` | 3D OBBs `[cx,cy,cz,xl,yl,zl,roll,pitch,yaw]` |
| `obj_tags` | `list[str]` | Object semantic labels |
| `axis_align_matrix` | `str` | Axis alignment matrix path |

### Per-scene record

Per-image fields are aggregated into lists, with `dataset` and `scene_id` kept as scalars. An additional `num_images` field records the view count.

## Supported Datasets

| Dataset | depth_scale | Annotations | Notes |
|---------|-------------|-------------|-------|
| ScanNet | 1000 | v1 | Images resized to match depth map dimensions |
| 3RScan | 1000 | v1 | Intrinsic parsed from `_info.txt` |
| Matterport3D | 4000 | v1 | Region-level scenes, complex camera naming |
| ARKitScenes | 1000 | v2 | Uses lowres_wide frames |
