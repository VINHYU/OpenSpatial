<p align="center">
  <img src="assets/logo.png" alt="OpenSpatial Logo" width="300">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.07296"><img src="https://img.shields.io/badge/arXiv-2604.07296-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/jdopensource/JoyAI-Image-OpenSpatial"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow" alt="Hugging Face"></a>
  <a href="https://huggingface.co/collections/VINHYU/openspatial-models-6a96dc5ba822865200e51e5c"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow" alt="Hugging Face Models"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache-2.0"></a>
</p>

**OpenSpatial** is an open-source 3D spatial understanding data engine engineered for **high quality**, **extensive scalability**, **broad task diversity**, and **optimized efficiency**. 

By bridging the gap between massive 2D web data and complex 3D spatial reasoning, OpenSpatial provides a comprehensive suite for the next generation of Embodied AI and World Models.

---

<p align="center">
  <img src="assets/teaser.png" alt="OpenSpatial Teaser" width="800">
  <br>
  <em>OpenSpatial Pipeline: From 2D Web Data to 3D Spatial Understanding</em>
</p>

---


## 🔥 News
- **[2026.09.02]** 🚀 We have released the **3D Lifting Module**, which reconstructs metric scene geometry and object-level 3D bounding boxes from ordered images or video. See the [3D Lifting documentation](#3d-lifting-module).
- **[2026.09.01]** 🤗 We have released the model weights for four **OpenSpatial spatial reasoning models**, covering InternVL2.5, InternVL3, Qwen2.5-VL, and Qwen3-VL backbones. Explore the [Hugging Face Collection](https://huggingface.co/collections/VINHYU/openspatial-models-6a96dc5ba822865200e51e5c) or see the [Model Zoo](#-model-zoo).
- **[2026.04.15]** 🎉 We have released the open-source subset of the **OpenSpatial-3M** dataset! Check it out on [Hugging Face](https://huggingface.co/datasets/jdopensource/JoyAI-Image-OpenSpatial).
- **[2026.04.08]** 🎉 The **OpenSpatial 3D data engine** is now officially open-sourced.

## 🚀 Key Features

* **Web Data 3D Lifting**: Advanced pipelines to transform large-scale 2D web imagery into geometrically consistent 3D representations.
* **Diverse Data Generation**: Automated engine for creating rich spatial understanding datasets, covering various environments and object-level details.
* **Multi-Task Integration**: Support for a wide range of tasks including 3D grounding, spatial reasoning, and scene captioning.
* **[Comprehensive Evaluation](https://github.com/open-compass/vlmevalkit)**: Built-in benchmarking suite to evaluate spatial understanding capabilities across different model architectures.
* **High Efficiency**: Optimized for large-scale data processing with scalable distributed computing support.

## 📊 Dataset

The **OpenSpatial-3M** dataset is now available on Hugging Face. It contains 3 million high-fidelity samples designed to enhance 3D spatial reasoning in large multi-modal models.

* **Repository**: [jdopensource/JoyAI-Image-OpenSpatial](https://huggingface.co/datasets/jdopensource/JoyAI-Image-OpenSpatial)

## 🤗 Model Zoo

We release four OpenSpatial checkpoints fine-tuned for spatial understanding and reasoning. The repositories contain full, inference-ready weights in `safetensors` format and are grouped in the [OpenSpatial Models Collection](https://huggingface.co/collections/VINHYU/openspatial-models-6a96dc5ba822865200e51e5c).

| Model | Base model | Checkpoint |
|---|---|---|
| OpenSpatial-InternVL2.5-8B | [OpenGVLab/InternVL2_5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B) | [Hugging Face](https://huggingface.co/VINHYU/OpenSpatial-InternVL2.5-8B) |
| OpenSpatial-InternVL3-8B | [OpenGVLab/InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B) | [Hugging Face](https://huggingface.co/VINHYU/OpenSpatial-InternVL3-8B) |
| OpenSpatial-Qwen2.5-VL-7B | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | [Hugging Face](https://huggingface.co/VINHYU/OpenSpatial-Qwen2.5-VL-7B) |
| OpenSpatial-Qwen3-VL-8B | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | [Hugging Face](https://huggingface.co/VINHYU/OpenSpatial-Qwen3-VL-8B) |

Download a checkpoint with the Hugging Face CLI:

```bash
hf download VINHYU/OpenSpatial-Qwen3-VL-8B \
  --local-dir OpenSpatial-Qwen3-VL-8B
```

See each model card for architecture-specific loading instructions and base-model requirements.

## 📖 Documentation

| Document | Description |
|---|---|
| [Quick Start](assets/quick_start.md) | Data preparation, config structure, annotation pipeline usage, and running tasks end-to-end |
| [Development Guide](assets/development_guide.md) | Adding new annotation tasks, pipeline stages, prompt templates, dataset preprocessors, and internal architecture reference |

## 3D Lifting Module

### Overview

The lifting module converts an ordered image sequence or video of one scene into metric depth, estimated cameras, a colored scene point cloud, and object-level 3D oriented bounding boxes. It preserves the released pipeline's Depth Anything 3 reconstruction, Gemini detection prompt, SAM 3 box-prompt segmentation, multi-view point aggregation, and floor-based Z-up alignment.

### Installation

Use a separate Python 3.12 environment. Install a PyTorch build suitable for your CUDA driver, the regular OpenSpatial requirements, and the optional lifting dependencies:

```bash
conda create -n openspatial-lifting python=3.12 -y
conda activate openspatial-lifting
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -r data_preprocessing/lifting/requirements.txt
```

Video input additionally requires `ffmpeg` on `PATH`. Object detection uses the public Gemini API and reads the credential from `GEMINI_API_KEY`; no API credential is included in this repository.

### Pretrained Model

The original implementation uses [Depth Anything 3 Nested Giant-Large](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE) and [SAM 3](https://huggingface.co/facebook/sam3). DA3's weights are CC BY-NC 4.0. SAM 3 uses Meta's SAM License and requires accepting its Hugging Face access request before download.

```bash
hf auth login
hf download depth-anything/DA3NESTED-GIANT-LARGE \
  --local-dir models/DA3NESTED-GIANT-LARGE
hf download facebook/sam3 sam3.pt --local-dir models/sam3
```

The semantic label matcher downloads `sentence-transformers/all-MiniLM-L6-v2` automatically. A local path may instead be supplied with `--semantic-model`.

### Input

Pass one image, a directory containing one scene's ordered `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp` images, or a video. Directory images are processed in lexical filename order. Video frames are extracted with `ffmpeg`; use `--fps` or `--max-frames` to limit them.

### Running

```bash
export GEMINI_API_KEY="<your-api-key>"
python -m data_preprocessing.lifting \
  --input /path/to/scene/images \
  --output outputs/lifting \
  --da3-model models/DA3NESTED-GIANT-LARGE \
  --sam3-checkpoint models/sam3/sam3.pt \
  --device cuda
```

The defaults preserve the original 504-pixel upper-bound resize, at most 100 reconstruction frames, and at most 80 detection views. These may be changed explicitly with `--process-resolution`, `--target-frames`, and `--detection-views`.

### Output

Each run creates `<output>/<scene>_<timestamp>/` containing:

- `openspatial.parquet`: one multiview row in OpenSpatial's dataset format;
- `pointclouds/scene_stitched.ply` and per-object `.ply` files in aligned world coordinates;
- `pointclouds/bounding_boxes.json`: object labels, descriptions, centers, sizes, and Z rotations;
- `processed_images/`, `depths/`, `masks/`, and `view_pointclouds/`: aligned per-view assets referenced by the Parquet file;
- `cameras/` and `camera_extrinsics.json`: camera-to-world text matrices and world-to-camera JSON matrices;
- `detections/`, `scene_metadata.json`, and `summary.json`: detector and run metadata.

Camera coordinates use OpenCV convention: X right, Y down, Z forward. Saved world geometry is metric and Z-up after floor alignment; the detected floor is shifted to Z=0. If no floor is detected, DA3 orientation is retained and the lowest reconstructed object point is shifted to Z=0. In `openspatial.parquet`, boxes use OpenSpatial's `[cx, cy, cz, xl, yl, zl, z_angle, 0, 0]` representation with `zxy` Euler order. Set an OpenSpatial config's `dataset.data_dir` to the generated Parquet file to use it in downstream multiview tasks.

### Example

For a small smoke run over four ordered frames:

```bash
python -m data_preprocessing.lifting \
  --input examples/room_frames \
  --output outputs/lifting \
  --da3-model models/DA3NESTED-GIANT-LARGE \
  --sam3-checkpoint models/sam3/sam3.pt \
  --device cuda --max-frames 4 --target-frames 4 --detection-views 4
```

## 📅 Roadmap & To-Do List

- [x] **3D Data Engine**: Open-source the core 3D spatial understanding data engine.
- [x] **OpenSpatial-3M Dataset Release**: Publicly release the large-scale 3M spatial understanding dataset. [[HF Link]](https://huggingface.co/datasets/jdopensource/JoyAI-Image-OpenSpatial)
- [x] **Model Weights Release**: Release four trained spatial understanding models with InternVL and Qwen-VL backbones. [[Model Zoo]](#-model-zoo) [[HF Collection]](https://huggingface.co/collections/VINHYU/openspatial-models-6a96dc5ba822865200e51e5c)
- [ ] **[Evaluation Suite](https://github.com/open-compass/vlmevalkit)**: Open-source the comprehensive evaluation code for spatial tasks.
- [x] **3D Lifting Module**: Release the core engine for lifting image sequences and video to metric 3D representations. [[Documentation]](#3d-lifting-module)
- [ ] **More Tasks**: Extend support for more spatial understanding task types.

## 📄 Citation

If you find OpenSpatial useful for your research, please consider citing our paper:

```bibtex
@article{openspatial2025,
  title={OpenSpatial: An Open-Source 3D Spatial Understanding Data Engine},
  journal={arXiv preprint arXiv:2604.07296},
  year={2025}
}
```
