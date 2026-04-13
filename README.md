<p align="center">
  <img src="assets/logo.png" alt="OpenSpatial Logo" width="300">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.07296"><img src="https://img.shields.io/badge/arXiv-2604.07296-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/jdopensource/JoyAI-Image-OpenSpatial"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow" alt="Hugging Face"></a>
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

> [!IMPORTANT]
> **We have open-sourced the 3D Data Engine and OpenSpatial-3M Dataset!** > Access the dataset here: [**JoyAI-Image-OpenSpatial**](https://huggingface.co/datasets/jdopensource/JoyAI-Image-OpenSpatial). Models will be released within the next two weeks.

---


## 🚀 Key Features

* **Web Data 3D Lifting**: Advanced pipelines to transform large-scale 2D web imagery into geometrically consistent 3D representations.
* **Diverse Data Generation**: Automated engine for creating rich spatial understanding datasets, covering various environments and object-level details.
* **Multi-Task Integration**: Support for a wide range of tasks including 3D grounding, spatial reasoning, and scene captioning.
* **Comprehensive Evaluation**: Built-in benchmarking suite to evaluate spatial understanding capabilities across different model architectures.
* **High Efficiency**: Optimized for large-scale data processing with scalable distributed computing support.

## 📊 Dataset

The **OpenSpatial-3M** dataset is now available on Hugging Face. It contains 3 million high-fidelity samples designed to enhance 3D spatial reasoning in large multi-modal models.

* **Repository**: [jdopensource/JoyAI-Image-OpenSpatial](https://huggingface.co/datasets/jdopensource/JoyAI-Image-OpenSpatial)

## 📖 Documentation

| Document | Description |
|---|---|
| [Quick Start](assets/quick_start.md) | Data preparation, config structure, annotation pipeline usage, and running tasks end-to-end |
| [Development Guide](assets/development_guide.md) | Adding new annotation tasks, pipeline stages, prompt templates, dataset preprocessors, and internal architecture reference |

## 📅 Roadmap & To-Do List

- [x] **3D Data Engine**: Open-source the core 3D spatial understanding data engine.
- [x] **OpenSpatial-3M Dataset Release**: Publicly release the large-scale 3M spatial understanding dataset. [[HF Link]](https://huggingface.co/datasets/jdopensource/JoyAI-Image-OpenSpatial)
- [ ] **Model Release**: Release the trained spatial understanding model.
- [ ] **Evaluation Suite**: Open-source the comprehensive evaluation code for spatial tasks.
- [ ] **3D Lifting Module**: Integrate the core engine for lifting 2D web data to 3D representations.
- [ ] **More Tasks**: Extend support for more spatial understanding task types.

## 📄 Citation

If you find OpenSpatial useful for your research, please consider citing our paper:

```bibtex
@article{openspatial2025,
  title={OpenSpatial: An Open-Source 3D Spatial Understanding Data Engine},
  journal={arXiv preprint arXiv:2604.07296},
  year={2025}
}