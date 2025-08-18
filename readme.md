<img src="imgs/nnInteractive_header_white.png">

# Python backend for `nnInteractive: Redefining 3D Promptable Segmentation`

This repository provides the official Python backend for `nnInteractive`, a state-of-the-art framework for 3D promptable segmentation. It is designed for seamless integration into Python-based workflows—ideal for researchers, developers, and power users working directly with code.

`nnInteractive` is also available through graphical viewers (GUI) for those who prefer a visual workflow. The napari and MITK integrations are developed and maintained by our team. Thanks to the community for contributing the 3D Slicer extension!


<div align="center">

| **<div align="center">[napari plugin](https://github.com/MIC-DKFZ/napari-nninteractive)</div>** | **<div align="center">[MITK integration](https://www.mitk.org/wiki/MITK-nnInteractive)</div>** | **<div align="center">[3D Slicer extension](https://github.com/coendevente/SlicerNNInteractive)</div>** | **<div align="center">[ITK-SNAP extension](https://itksnap-dls.readthedocs.io/en/latest/quick_start.html)</div>** |
|-------------------|----------------------|-------------------------|-------------------------|
| [<img src="imgs/Logos/napari.jpg" height="200">](https://github.com/MIC-DKFZ/napari-nninteractive) | [<img src="imgs/Logos/mitk.jpg" height="200">](https://www.mitk.org/wiki/MITK-nnInteractive) | [<img src="imgs/Logos/3DSlicer.png" height="200">](https://github.com/coendevente/SlicerNNInteractive) | [<img src="imgs/Logos/snaplogo_sq.png" height="200">](https://itksnap-dls.readthedocs.io/en/latest/quick_start.html) |

</div>


## 📰 News

- **07/2025**: 🧩 New ITK-SNAP extension released! Try nnInteractive directly in ITK-SNAP 👉 [Quick Start](https://itksnap-dls.readthedocs.io/en/latest/quick_start.html)
- **06/2025**: 🏆 We’re thrilled to announce that `nnInteractive` **won the 1st place** in the [CVPR 2025 Challenge on Interactive 3D Segmentation](https://www.codabench.org/competitions/5263/). Huge shoutout to the organizers and all contributors!
- **05/2025**: `nnInteractive` presents an **official baseline** at **CVPR 2025** in the _Foundation Models for Interactive 3D Biomedical Image Segmentation Challenge_ ([Codabench link](https://www.codabench.org/competitions/5263/)) → see [`nnInteractive/inference/cvpr2025_challenge_baseline`](nnInteractive/inference/cvpr2025_challenge_baseline)
- **04/2025**: 🎉 The **community contributed a 3D Slicer integration** – thank you! 👉 [SlicerNNInteractive](https://github.com/coendevente/SlicerNNInteractive)
- **03/2025**: 🚀 `nnInteractive` **launched** with native support for **napari** and **MITK**

---


## What is nnInteractive?

> Isensee, F.\*, Rokuss, M.\*, Krämer, L.\*, Dinkelacker, S., Ravindran, A., Stritzke, F., Hamm, B., Wald, T., Langenberg, M., Ulrich, C., Deissler, J., Floca, R., & Maier-Hein, K. (2025). nnInteractive: Redefining 3D Promptable Segmentation. https://arxiv.org/abs/2503.08373 \
> *: equal contribution

Link: [![arXiv](https://img.shields.io/badge/arXiv-2503.08373-b31b1b.svg)](https://arxiv.org/abs/2503.08373)


##### Abstract:

Accurate and efficient 3D segmentation is essential for both clinical and research applications. While foundation 
models like SAM have revolutionized interactive segmentation, their 2D design and domain shift limitations make them 
ill-suited for 3D medical images. Current adaptations address some of these challenges but remain limited, either 
lacking volumetric awareness, offering restricted interactivity, or supporting only a small set of structures and 
modalities. Usability also remains a challenge, as current tools are rarely integrated into established imaging 
platforms and often rely on cumbersome web-based interfaces with restricted functionality. We introduce nnInteractive, 
the first comprehensive 3D interactive open-set segmentation method. It supports diverse prompts—including points, 
scribbles, boxes, and a novel lasso prompt—while leveraging intuitive 2D interactions to generate full 3D 
segmentations. Trained on 120+ diverse volumetric 3D datasets (CT, MRI, PET, 3D Microscopy, etc.), nnInteractive 
sets a new state-of-the-art in accuracy, adaptability, and usability. Crucially, it is the first method integrated 
into widely used image viewers (e.g., Napari, MITK), ensuring broad accessibility for real-world clinical and research 
applications. Extensive benchmarking demonstrates that nnInteractive far surpasses existing methods, setting a new 
standard for AI-driven interactive 3D segmentation.

<img src="imgs/figure1_method.png" width="1200">


## Installation

### Prerequisites

You need a Linux or Windows computer with a Nvidia GPU. 10GB of VRAM is recommended. Small objects should work with \<6GB.

##### 1. Create a virtual environment:

nnInteractive supports Python 3.10+ and works with Conda, pip, or any other virtual environment. Here’s an example using Conda:

```
conda create -n nnInteractive python=3.12
conda activate nnInteractive
```

##### 2. Install the correct PyTorch for your system

Go to the [PyTorch homepage](https://pytorch.org/get-started/locally/) and pick the right configuration.
Note that since recently PyTorch needs to be installed via pip. This is fine to do within your conda environment.

For Ubuntu with a Nvidia GPU, pick 'stable', 'Linux', 'Pip', 'Python', 'CUDA12.6' (if all drivers are up to date, otherwise use and older version):

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

##### 3. Install this repository
Either install via pip:
`pip install nninteractive`

Or clone and install this repository:
```bash
git clone https://github.com/MIC-DKFZ/nnInteractive
cd nnInteractive
pip install -e .
```

## Getting Started
Here is a minimalistic script that covers the core functionality of nnInteractive:

```python
import os
import torch
import SimpleITK as sitk
from huggingface_hub import snapshot_download  # Install huggingface_hub if not already installed

# --- Download Trained Model Weights (~400MB) ---
REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future
DOWNLOAD_DIR = "/home/isensee/temp"  # Specify the download directory

download_path = snapshot_download(
    repo_id=REPO_ID,
    allow_patterns=[f"{MODEL_NAME}/*"],
    local_dir=DOWNLOAD_DIR
)

# The model is now stored in DOWNLOAD_DIR/MODEL_NAME.

# --- Initialize Inference Session ---
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

session = nnInteractiveInferenceSession(
    device=torch.device("cuda:0"),  # Set inference device
    use_torch_compile=False,  # Experimental: Not tested yet
    verbose=False,
    torch_n_threads=os.cpu_count(),  # Use available CPU cores
    do_autozoom=True,  # Enables AutoZoom for better patching
    use_pinned_memory=True,  # Optimizes GPU memory transfers
)

# Load the trained model
model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
session.initialize_from_trained_model_folder(model_path)

# --- Load Input Image (Example with SimpleITK) ---
# DO NOT preprocess the image in any way. Give it to nnInteractive as it is! DO NOT apply level window, DO NOT normalize 
# intensities and never ever convert an image with higher precision (float32, uint16, etc) to uint8!
# The ONLY instance where some preprocesing makes sense is if your original image is too large to be reasonably used. 
# This may be the case, for example, for some microCT images. In this case you can consider downsampling.
input_image = sitk.ReadImage("FILENAME")
img = sitk.GetArrayFromImage(input_image)[None]  # Ensure shape (1, x, y, z)

# Validate input dimensions
if img.ndim != 4:
    raise ValueError("Input image must be 4D with shape (1, x, y, z)")

session.set_image(img)

# --- Define Output Buffer ---
target_tensor = torch.zeros(img.shape[1:], dtype=torch.uint8)  # Must be 3D (x, y, z)
session.set_target_buffer(target_tensor)

# --- Interacting with the Model ---
# Interactions can be freely chained and mixed in any order. Each interaction refines the segmentation.
# The model updates the segmentation mask in the target buffer after every interaction.

# Example: Add a **positive** point interaction
# POINT_COORDINATES should be a tuple (x, y, z) specifying the point location.
session.add_point_interaction(POINT_COORDINATES, include_interaction=True)

# Example: Add a **negative** point interaction
# To make any interaction negative set include_interaction=False
session.add_point_interaction(POINT_COORDINATES, include_interaction=False)

# Example: Add a bounding box interaction
# BBOX_COORDINATES must be specified as [[x1, x2], [y1, y2], [z1, z2]] (half-open intervals).
# Note: nnInteractive pre-trained models currently only support **2D bounding boxes**.
# This means that **one dimension must be [d, d+1]** to indicate a single slice.

# Example of a 2D bounding box in the axial plane (XY slice at depth Z)
# BBOX_COORDINATES = [[30, 80], [40, 100], [10, 11]]  # X: 30-80, Y: 40-100, Z: slice 10

session.add_bbox_interaction(BBOX_COORDINATES, include_interaction=True)

# Example: Add a scribble interaction
# - A 3D image of the same shape as img where one slice (any axis-aligned orientation) contains a hand-drawn scribble.
# - Background must be 0, and scribble must be 1.
# - Use session.preferred_scribble_thickness for optimal results.
session.add_scribble_interaction(SCRIBBLE_IMAGE, include_interaction=True)

# Example: Add a lasso interaction
# - Similarly to scribble a 3D image with a single slice containing a **closed contour** representing the selection.
session.add_lasso_interaction(LASSO_IMAGE, include_interaction=True)

# You can combine any number of interactions as needed. 
# The model refines the segmentation result incrementally with each new interaction.

# --- Retrieve Results ---
# The target buffer holds the segmentation result.
results = session.target_buffer.clone()
# OR (equivalent)
results = target_tensor.clone()

# Cloning is required because the buffer will be **reused** for the next object.
# Alternatively, set a new target buffer for each object:
session.set_target_buffer(torch.zeros(img.shape[1:], dtype=torch.uint8))

# --- Start a New Object Segmentation ---
session.reset_interactions()  # Clears the target buffer and resets interactions

# Now you can start segmenting the next object in the image.

# --- Set a New Image ---
# Setting a new image also requires setting a new matching target buffer
session.set_image(NEW_IMAGE)
session.set_target_buffer(torch.zeros(NEW_IMAGE.shape[1:], dtype=torch.uint8))

# Enjoy!
```

## nnInteractive SuperVoxels

As part of the `nnInteractive` framework, we provide a dedicated module for **supervoxel generation** based on [SAM](https://github.com/facebookresearch/segment-anything) and [SAM2](https://github.com/facebookresearch/sam2). This replaces traditional superpixel methods (e.g., SLIC) with **foundation model–powered 3D pseudo-labels**.

🔗 **Module:** [`nnInteractive/supervoxel/`](nnInteractive/supervoxel)

The SuperVoxel module allows you to:

- Automatically generate high-quality 3D supervoxels via axial sampling + SAM segmentation and SAM2 mask propagation.
- Use the generated supervoxels as **pseudo-ground-truth labels** to train promptable 3D segmentation models like `nnInteractive`.
- Export `nnUNet`-compatible `.pkl` foreground prompts for downstream use.

For detailed installation, configuration, and usage instructions, check the [SuperVoxel README](nnInteractive/supervoxel/README.md).


## Citation
When using nnInteractive, please cite the following paper:

> Isensee, F.\*, Rokuss, M.\*, Krämer, L.\*, Dinkelacker, S., Ravindran, A., Stritzke, F., Hamm, B., Wald, T., Langenberg, M., Ulrich, C., Deissler, J., Floca, R., & Maier-Hein, K. (2025). nnInteractive: Redefining 3D Promptable Segmentation. https://arxiv.org/abs/2503.08373 \
> *: equal contribution

Link: [![arXiv](https://img.shields.io/badge/arXiv-2503.08373-b31b1b.svg)](https://arxiv.org/abs/2503.08373)


# License
Note that while this repository is available under Apache-2.0 license (see [LICENSE](./LICENSE)), the [model checkpoint](https://huggingface.co/nnInteractive/nnInteractive) is `Creative Commons Attribution Non Commercial Share Alike 4.0`! 

# Changelog

### 1.1.2 - 2025-08-02

- Fixed a bug where `pin_memory` was set to `True` even though no CUDA devices were present (this broke CPU support)
- ✅ API compatible all the way back to 1.0.1

### 1.1.1 - 2025-08-01

- We now detect whether linux kernel 6.11 is used and disable pin_memory in that case. See also [here](https://github.com/MIC-DKFZ/nnInteractive/issues/18)
- ✅ API compatible with 1.0.1 and 1.1.0

### 1.1.0 - 2025-08-01

- Reworked inference code. It's now well-structured and easier to follow.
- Fixed bugs that 
  - sometimes caused blocky predictions
  - may cause failure to update segmentation map if changes were minor and AutoZoom was triggered
- ✅ API compatible with 1.0.1

## Acknowledgments

<p align="left">
  <img src="imgs/Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="imgs/Logos/DKFZ_Logo.png" width="500">
</p>

This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the 
[Division of Medical Image Computing](https://www.dkfz.de/en/medical-image-computing) at DKFZ.
