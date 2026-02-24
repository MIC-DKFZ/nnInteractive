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

Session-based inference engine (~805 lines) that manages state across multiple predictions. This is the main user-facing API.

**Workflow**: `set_image()` → `add_*_interaction()` → `predict()` (repeatable)

- **Background threading**: Image preprocessing and interaction initialization run in a ThreadPoolExecutor (max 2 workers) via futures
- **AutoZoom**: Adaptive patch selection with border change detection; zooms out up to 4x if predictions touch crop boundaries
- **Refinement**: After coarse prediction, difference maps identify regions needing fine-grained re-prediction
- **Memory management**: Selective CPU/GPU transfers, pre-allocated tensors, half-precision interactions, pinned memory (disabled on Linux kernel 6.11)

### 7-Channel Interaction Tensor Layout

The interaction tensor has shape `[7, D, H, W]` in half precision:
| Channel | Content |
|---------|---------|
| 0 | Initial segmentation |
| 1 | Positive bounding box / Lasso |
| 2 | Negative bounding box / Lasso |
| 3 | Positive points |
| 4 | Negative points |
| 5 | Positive scribble |
| 6 | Negative scribble |

### Key Modules

- **`nnInteractive/interaction/point.py`**: Point interaction with spherical structuring elements and distance transforms. Uses `lru_cache` for structuring element reuse.
- **`nnInteractive/trainer/nnInteractiveTrainer.py`**: Minimal stub that extends nnUNetv2 to add 7 interaction input channels. Used only for architecture reconstruction from checkpoints.
- **`nnInteractive/utils/crop.py`**: Tensor cropping/padding with boundary handling (`crop_and_pad_into_buffer`, `paste_tensor`)
- **`nnInteractive/utils/bboxes.py`**: Greedy set cover algorithm for generating refinement patch bounding boxes from difference maps
- **`nnInteractive/supervoxel/`**: Optional separate module for SAM-based supervoxel generation (has its own `pyproject.toml` and installation)

### Key Dependencies

- **`nnunetv2`** (>=2.6): Provides network architecture, ConfigurationManager, PlansManager, preprocessing pipeline
- **`torch`** (>=2.6, <2.9.0): PyTorch 2.9.0 is excluded due to OOM bugs with 3D convolutions
- **`acvl-utils`** (>=0.2.3, <0.3): Spatial operations (cropping, padding)

### Coordinate System

Images are 4D numpy arrays `[C, X, Y, Z]`. During preprocessing, images are cropped to their nonzero region (`bbox_used_for_cropping`). All interaction coordinates must be transformed between original and cropped space via `transform_coordinates_noresampling()`.

### Platform Workarounds

- Linux kernel 6.11 detection (`utils/os_shennanigans.py`) disables pinned memory due to a kernel bug
- `interaction_decay` (0.9 or 0.98) downweights older interactions on each prediction cycle
