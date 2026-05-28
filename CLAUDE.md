# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nnInteractive is a 3D interactive medical image segmentation framework. It supports diverse prompt types (points, scribbles, bounding boxes, lasso) using 2D interactions to generate full 3D segmentations. This repository contains the **inference-only** package.

## Build & Development Commands

```bash
# Install from source (editable)
pip install -e .

# Install with dev tools
pip install -e ".[dev]"

# Code formatting
black nnInteractive/

# Linting
ruff check nnInteractive/ --fix

# Spell checking
codespell --skip='.git,*.pdf,*.svg'

# Pre-commit hooks (after: pre-commit install)
pre-commit run --all-files
```

There is no test suite in this repository.

## Architecture

### Core Class: `nnInteractiveInferenceSession` (`nnInteractive/inference/inference_session.py`)

Session-based inference engine (~1040 lines) that manages state across multiple predictions. This is the main user-facing API.

**Workflow**: `initialize_from_trained_model_folder()` → `set_image()` → `set_target_buffer()` → `add_*_interaction()` (repeatable)

- **Background threading**: Image preprocessing and interaction initialization run in a ThreadPoolExecutor (max 2 workers) via futures
- **AutoZoom**: Adaptive patch selection with border change detection; zooms out up to 4x if predictions touch crop boundaries
- **Refinement**: After coarse prediction, difference maps identify regions needing fine-grained re-prediction
- **Memory management**: Selective CPU/GPU transfers, pre-allocated tensors, half-precision interactions, pinned memory (disabled on Linux kernel 6.11)

**Important API constraints**:
- `use_torch_compile=False` is required (there is an assertion; `torch.compile` is incompatible with pinned memory)
- `target_buffer` must be 3D (shape `[X, Y, Z]`), not 4D
- Scribble and lasso images must match `original_image_shape[1:]` (the original uncropped spatial shape)
- `add_initial_seg_interaction()` **resets all existing interactions** (see WARNING in its docstring)
- `reset_interactions()` also clears the target buffer
- If multiple `add_*_interaction()` calls are made without calling `_predict()`, only the last added interaction center is used for the initial prediction (all centers are queued, but only the last is consumed)
- Images should **not** be preprocessed (no normalization, no level-window). The session handles all preprocessing internally

### 7-Channel Interaction Tensor Layout

The interaction tensor has shape `[7, D, H, W]` in half precision (but can vary — `num_interaction_channels` is set from `capability['interaction_channels']`):
| Channel | Content |
|---------|---------|
| 0 | Initial segmentation |
| 1 | Positive bounding box / Lasso |
| 2 | Negative bounding box / Lasso |
| 3 | Positive points |
| 4 | Negative points |
| 5 | Positive scribble |
| 6 | Negative scribble |

### Capability / Channel Mapping System

`initialize_from_trained_model_folder()` reads model metadata from:
1. `inference_info.json` (new format) — contains `supported_interactions`, `channel_mapping`, `interaction_channels`, `interaction_decay`, `point_radius`, etc.
2. `inference_session_class.json` (legacy format) — falls back to hardcoded defaults

The capability system (`_apply_capability()`) normalizes all channel indices to positive values at load time. Channel mappings use pairs `[pos_channel, neg_channel]` for interactions and a single index for `prev_seg`.

### Model Checkpoint Structure

A trained model folder must contain:
- `inference_info.json` (or legacy `inference_session_class.json`)
- `dataset.json`
- `plans.json`
- `fold_{N}/checkpoint_final.pth` (or `fold_all/` for ensemble)

Official weights are hosted on HuggingFace at `nnInteractive/nnInteractive`, model name `nnInteractive_v1.0`.

### Key Modules

- **`nnInteractive/interaction/point.py`**: Point interaction with spherical structuring elements and distance transforms. `build_point()` uses `lru_cache` for structuring element reuse.
- **`nnInteractive/trainer/nnInteractiveTrainer.py`**: Minimal stub extending nnUNetv2. Used only for architecture reconstruction from checkpoints. Adds 7 extra input channels (`num_input_channels + 7`) on top of image channels.
- **`nnInteractive/utils/crop.py`**: Tensor cropping/padding with boundary handling (`crop_and_pad_into_buffer`, `paste_tensor`, `crop_to_valid`, `pad_cropped`)
- **`nnInteractive/utils/bboxes.py`**: Greedy set cover algorithm for generating refinement patch bounding boxes from difference maps; falls back to random sampling when recursion depth exceeds `max_depth`
- **`nnInteractive/utils/erosion_dilation.py`**: `iterative_3x3_same_padding_pool3d` — used for dilation of point/scribble channels before downsampling (zoom-out) and for morphological opening of the diff map
- **`nnInteractive/utils/checkpoint_cleansing.py`**: Utility to strip optimizer state and trainer class from checkpoints before release
- **`nnInteractive/inference/cvpr2025_challenge_baseline/predict.py`**: Reference script for the CVPR 2025 challenge baseline (stateless per-call API wrapper around `nnInteractiveInferenceSession`)
- **`nnInteractive/supervoxel/`**: Optional separate module for SAM-based supervoxel generation (has its own `pyproject.toml` and installation)

### Key Dependencies

- **`nnunetv2`** (>=2.6): Provides network architecture, ConfigurationManager, PlansManager, preprocessing pipeline
- **`torch`** (>=2.6, <2.9.0): PyTorch 2.9.0 is excluded due to OOM bugs with 3D convolutions
- **`acvl-utils`** (>=0.2.3, <0.3): Spatial operations (cropping, padding)

### Coordinate System

Images are 4D numpy arrays `[C, X, Y, Z]`. During preprocessing, images are cropped to their nonzero region (`bbox_used_for_cropping`). All interaction coordinates must be transformed between original and cropped space via `transform_coordinates_noresampling()`.

Bounding box coordinates use `[[x1, x2], [y1, y2], [z1, z2]]` half-open intervals throughout. Current pretrained models only support **2D bounding boxes** (one dimension must have size 1).

### `_predict()` Implementation Notes

The `_predict()` method (decorated with `@torch.inference_mode()`) is highly optimized for minimal VRAM usage. Comments in the code note that the implementation has been extensively tuned and changes should only be made after fully understanding the VRAM/timing implications. The method:
1. Runs an initial coarse prediction at `zoom_out_factor`
2. Detects changes at prediction borders (`_detect_change_at_border`) and iteratively zooms out (up to 4x, growth factor 1.5) if needed
3. If `zoom_out_factor == 1`: directly pastes prediction into interactions and target buffer
4. If `zoom_out_factor > 1`: computes a diff map, morphologically opens it, then runs refinement patches via `_refine_coarse()`

### Platform Workarounds

- Linux kernel 6.11 detection (`utils/os_shennanigans.py`) disables pinned memory due to a kernel bug
- `interaction_decay` (default 0.98, legacy 0.9) downweights older interaction channels (all except `prev_seg`) on each prediction cycle
