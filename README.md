<div align="center" markdown="1">

<img src="https://github.com/user-attachments/assets/1935c89a-76cb-4edc-a6ac-0c3658d347f6" alt="openworldlib_logo" width="90%" />

#### Welcome to join us open-source world model project ! <!-- omit in toc -->
---

<a href="https://github.com/OpenDCAI/OpenWorldLib"><img alt="Build" src="https://img.shields.io/github/stars/OpenDCAI/OpenWorldLib"></a> <!-- License --> <a href="https://github.com/OpenDCAI/OpenWorldLib/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/OpenDCAI/OpenWorldLib"></a> <!-- GitHub Issues --> <a href="https://github.com/OpenDCAI/OpenWorldLib/issues"><img alt="Issues" src="https://img.shields.io/github/issues/OpenDCAI/OpenWorldLib"></a>
<img alt="Report" src="https://img.shields.io/badge/📄 Technical Report-Coming Soon-lightgrey">
<!-- <a href="https://arxiv.org/abs/xxxx.xxxxx"><img alt="Paper" src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b?logo=arxiv&logoColor=white"></a> -->

[English](README.md) | [中文](README_zh.md)

Extension repo：[[3D generation]](https://github.com/zengbohan0217/OpenWorldLib-extension-3D) | [[VLA]]() | [[simulator]]()

</div>


We define a world model as: **A model or framework centered on perception, equipped with interaction and long-term memory capabilities, for understanding and predicting the complex world.** Accordingly, 🎓 *Multimodal Understanding*, 🤖 *Visual Action Prediction*, and 🖼️ *Visual Generation* are all sub-tasks that a world model needs to accomplish.

We warmly welcome researchers to share their views on this framework or thoughts on world models in the Issues section. We also hope that you can submit valuable world-model-related methods to our framework via Pull Requests, or document and submit them to [[awesome_world_models]](docs/awesome_world_model.md). Feel free to give our repo a star 🌟 to follow the latest progress of OpenWorldLib!

<!-- the demonstration demo insert here -->

### Important Docs <!-- omit in toc -->
The following three documents are essential to this project (click to navigate):

- [docs/planning.md](docs/planning.md): This document tracks the short-term optimization goals and future development plans for OpenWorldLib.
- [docs/awesome_world_models.md](docs/awesome_world_model.md): This document records cutting-edge research, related surveys, and open-source projects on world models.
- [docs/installation.md](docs/installation.md): This document provides installation instructions for different methods in OpenWorldLib.


### Table of Contents <!-- omit in toc -->
- [Features](#features)
  - [Project Goals](#project-goals)
  - [Supported Tasks](#supported-tasks)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
- [Structure](#structure)
- [Planning](#planning)
- [For Developers](#for-developers)
- [Acknowledgment](#acknowledgment)
- [Citation](#citation)


### Features
#### Project Goals
The main goals of OpenWorldLib include:
- Establishing a unified and standardized **world model framework** to make the invocation of existing world-model-related code more consistent and well-structured;
- Integrating open-source world model research outcomes and systematically curating related papers for researchers' reference and use.

#### Supported Tasks
OpenWorldLib covers the following research directions related to **World Models**:

| Task Category | Sub-task | Representative Methods/Models |
| :--- | :--- | :--- |
| **Video Generation** | Navigation Generation | lingbot, matrix-game, hunyuan-worldplay, genie3, etc. |
| | Long Video Generation | sora-2, veo-3, wan, etc. |
| **3D Scene Generation** | 3D Scene Generation | flash-world, vggt, etc. |
| **Reasoning** | VQA (Visual Question Answering) | spatialVLM, omnivinci and other VLMs with world understanding |
| | VLA (Vision-Language-Action) | pi-0, pi-0.5, giga-brain, etc. |


### Getting Started
#### Installation
First, create a conda environment:
```bash
conda create -n "openworldlib" python=3.10 -y
conda activate "openworldlib"
```
Then install using the provided script:
```bash
cd OpenWorldLib
bash scripts/setup/default_install.sh
```
Some methods have special installation requirements. All installation scripts are located in `./scripts/setup`.
> 📖 For the full installation guide, please refer to [docs/installation.md](docs/installation.md)

#### Quickstart
After installing the base environment, you can test matrix-game-2 generation and multi-turn interaction with the following commands:
```bash
cd OpenWorldLib
bash scripts/test_inference/test_nav_video_gen.sh matrix-game-2
bash scripts/test_stream/test_nav_video_gen.sh matrix-game-2
```
Scripts for other methods can be found under `scripts/test_inference` and `scripts/test_stream`.

### Structure
To help developers and users better understand OpenWorldLib, we provide details about our codebase. The framework structure is as follows:
```txt
OpenWorldLib
├─ assets
├─ data                                # Test data
│  ├─ benchmarks
│  │  └─ reasoning
│  ├─ test_case
│  └─ ...
├─ docs                                # Documentation
├─ examples                            # Benchmark examples
├─ scripts                             # All key test scripts
├─ src
│  └─ openworldlib                        # Main source path
│     ├─ base_models                   # Base models
│     │  ├─ diffusion_model
│     │  │  ├─ image
│     │  │  ├─ video
│     │  │  └─ ...
│     │  ├─ llm_mllm_core
│     │  │  ├─ llm
│     │  │  ├─ mllm
│     │  │  └─ ...
│     │  ├─ perception_core
│     │  │  ├─ detection
│     │  │  ├─ general_perception
│     │  │  └─ ...
│     │  └─ three_dimensions
│     │     ├─ depth
│     │     ├─ general_3d
│     │     └─ ...
│     ├─ memories                      # Memory module
│     │  ├─ reasoning
│     │  └─ visual_synthesis
│     ├─ operators                     # Input & interaction signal processing
│     ├─ pipelines                     # All runtime pipelines
│     ├─ reasoning                     # Reasoning module
│     │  ├─ audio_reasoning
│     │  ├─ general_reasoning
│     │  └─ spatial_reasoning
│     ├─ representations               # Representation module
│     │  ├─ point_clouds_generation
│     │  └─ simulation_environment
│     └─ synthesis                     # Generation module
│        ├─ audio_generation
│        ├─ visual_generation
│        └─ vla_generation
├─ submodules                          # Auxiliary installs (e.g., diff-gaussian-raster)
├─ test                                # All test code
├─ test_stream                         # All interactive test code
└─ tools                               # Utilities
   ├─ installing
   └─ vibe_code
```
When using OpenWorldLib, users typically call the **pipeline** class directly, which handles weight loading, environment initialization, and other tasks. Users interact with the **operator** class, and leverage the **synthesis**, **reasoning**, and **representation** classes for generation. In multi-turn interactions, the **memory** class is used to maintain the running context.

### Planning

- We document the latest cutting-edge world model research in docs/awesome_world_models.md, and welcome contributions of valuable research.
- We document our upcoming training and optimization plans in docs/planning.md.


### For Developers
We welcome all developers to contribute and help improve **OpenWorldLib** as a unified world model repository. We recommend using **Vibe Coding** for quick code contributions — related prompts can be found under `tools/vibe_code/prompts`. You are also encouraged to add high-quality world model works to [docs/planning.md](docs/planning.md) and [docs/awesome_world_models.md](docs/awesome_world_model.md). We look forward to your contributions!

Related documents: [[Project Overview]](https://wcny4qa9krto.feishu.cn/wiki/NAy4wGbGrilzZ6kVInKcqkB9nUe) | [[Development Guide]](https://wcny4qa9krto.feishu.cn/wiki/XF2Ew583ziTT8LkNiAQcyZtTn2N) | [[Task Assignment Details]](https://wcny4qa9krto.feishu.cn/wiki/NLFdw1NpBickgCkb2WQcGJrInUe) | [[Code Submission Guidelines]](https://wcny4qa9krto.feishu.cn/wiki/Zs8cwPWqMi45FSknzLPc1WlhnAg)


### Acknowledgment
This project is an extension of DataFlow and DataFlow-MM for world model tasks. We are also actively collaborating with RayOrch, Paper2Any, and other projects.


### Citation
If OpenWorldLib has been helpful to you, please consider giving our repo a star 🌟 and citing the related papers:
```bibtex
@misc{dataflow-team-openworldlib,
  author = {{OpenDCAI}},
  title = {OpenWorldLib: A Unified Codebase for Advanced World Models},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/OpenDCAI/OpenWorldLib}}
}

@article{zeng2026research,
  title={Research on World Models Is Not Merely Injecting World Knowledge into Specific Tasks},
  author={Zeng, Bohan and Zhu, Kaixin and Hua, Daili and Li, Bozhou and Tong, Chengzhuo and Wang, Yuran and Huang, Xinyi and Dai, Yifan and Zhang, Zixiang and Yang, Yifan and others},
  journal={arXiv preprint arXiv:2602.01630},
  year={2026}
}
```
To further elaborate on our framework's design philosophy and our understanding of world models, we will release a technical report for OpenWorldLib. We hope our work is helpful to you!
