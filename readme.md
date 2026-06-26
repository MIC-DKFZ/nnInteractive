<img src="imgs/nnInteractive_header_white.png">


# Python backend for `nnInteractive: Redefining 3D Promptable Segmentation`

This repository provides the official Python backend for `nnInteractive`, a state-of-the-art framework for 3D promptable segmentation. It is designed for seamless integration into Python-based workflows—ideal for researchers, developers, and power users working directly with code.

`nnInteractive` is also available through graphical viewers (GUI) for those who prefer a visual workflow.

### Recommended integrations (developed and maintained by our team)

<div align="center">

| **<div align="center">[napari plugin](https://github.com/MIC-DKFZ/napari-nninteractive)</div>** | **<div align="center">[MITK integration](https://www.mitk.org/)</div>** |
|-------------------|----------------------|
| [<img src="imgs/Logos/napari.jpg" height="200">](https://github.com/MIC-DKFZ/napari-nninteractive) | [<img src="imgs/Logos/mitk.jpg" height="200">](https://www.mitk.org/) |

</div>

### Community-driven integrations

Huge thanks to the community for contributing these integrations!

<div align="center">

| **<div align="center">[3D Slicer extension](https://github.com/coendevente/SlicerNNInteractive)</div>** | **<div align="center">[ITK-SNAP extension](https://itksnap-dls.readthedocs.io/en/latest/quick_start.html)</div>** | **<div align="center">[OHIF integration](https://github.com/CCI-Bonn/OHIF-AI)</div>** |
|-------------------------|-------------------------|-------------------------|
| [<img src="imgs/Logos/3DSlicer.png" height="200">](https://github.com/coendevente/SlicerNNInteractive) | [<img src="imgs/Logos/snaplogo_sq.png" height="200">](https://itksnap-dls.readthedocs.io/en/latest/quick_start.html) | [<img src="imgs/Logos/ohif.png" height="200">](https://github.com/CCI-Bonn/OHIF-AI) |

</div>


## 📰 News

- **06/2026**: 🐳 The inference server now ships as a **Docker image** — run nnInteractive on a GPU box with a single `docker run`, no install required. See [`DOCKER.md`](nnInteractive/inference/server/DOCKER.md)
- **05/2026**: 🌐 **Remote GPU inference** (client/server): drive nnInteractive over HTTP from a machine without a GPU via `nnInteractiveRemoteInferenceSession`, with one server hosting multiple concurrent sessions. See [`SERVER_CLIENT.md`](SERVER_CLIENT.md)
- **11/2025**: 🌐 New community driven **OHIF integration** released by our colleagues at [CCI Bonn](https://ccibonn.ai/). Try nnInteractive directly in OHIF 👉 [OHIF-AI](https://github.com/CCI-Bonn/OHIF-AI)
- **07/2025**: 🧩 New ITK-SNAP extension released! Try nnInteractive directly in ITK-SNAP 👉 [Quick Start](https://itksnap-dls.readthedocs.io/en/latest/quick_start.html)
- **06/2025**: 🏆 We’re thrilled to announce that `nnInteractive` **won the 1st place** in the [CVPR 2025 Challenge on Interactive 3D Segmentation](https://www.codabench.org/competitions/5263/). Huge shoutout to the organizers and all contributors!
- **05/2025**: `nnInteractive` presents an official baseline at CVPR 2025 in the _Foundation Models for Interactive 3D Biomedical Image Segmentation Challenge_ ([Codabench link](https://www.codabench.org/competitions/5263/)) → see [`nnInteractive/inference/cvpr2025_challenge_baseline`](nnInteractive/inference/cvpr2025_challenge_baseline)
- **04/2025**: 🎉 The community contributed a 3D Slicer integration – thank you! 👉 [SlicerNNInteractive](https://github.com/coendevente/SlicerNNInteractive)
- **03/2025**: 🚀 `nnInteractive` launched with native support for napari and MITK

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

nnInteractive ships as **two pip packages — install only what you need:**

- **`nninteractive-client`** — lightweight remote client that drives a remote
  `nninteractive-server` (via `nnInteractiveRemoteInferenceSession`). **No PyTorch, no GPU** —
  only `numpy` / `httpx` / `blosc2`. Ideal for a GUI or thin client.
- **`nnInteractive`** — the full stack: the local inference engine *and* the official
  server. Needs **PyTorch and an NVIDIA GPU** (10 GB VRAM recommended; small objects work with
  \<6 GB). It depends on `nninteractive-client`, so it includes the remote client too.

Both expose the same `nnInteractive` import namespace, so client code is identical either way.

##### 1. Create a virtual environment

nnInteractive supports Python 3.10+ and works with Conda, pip, or any other virtual environment. Here’s an example using Conda:

```
conda create -n nnInteractive python=3.12
conda activate nnInteractive
```

##### 2a. Lightweight remote client (no PyTorch, no GPU)

If this machine only needs to *talk to* a remote `nninteractive-server`, install just the client:

```bash
pip install nninteractive-client
```

That's it — no PyTorch step required. You can upgrade to the full stack later with
`pip install nnInteractive` (no uninstall needed); using a full-only feature (local inference,
the server) from a client-only install raises a clear error telling you to do so.

##### 2b. Full stack (local inference + server, needs an NVIDIA GPU)

**First** install the correct PyTorch for your system — PyTorch is only needed for this full
install. Go to the [PyTorch homepage](https://pytorch.org/get-started/locally/) and pick the
right configuration. For Ubuntu with an NVIDIA GPU and up-to-date drivers, pick 'stable',
'Linux', 'Pip', 'Python', 'CUDA 12.6' (use an older CUDA if your drivers are older):

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**Then** install nnInteractive (this also pulls in the remote client):

```bash
pip install nnInteractive
```

##### Editable / development install (from source)

This repository builds **two distributions that share the `nnInteractive` import namespace**
(via [PEP 420 namespace packages](https://peps.python.org/pep-0420/)):

- **`nninteractive-client`** — the torch-free remote client (`nnInteractive.inference.remote`),
  with its source under `client/`;
- **`nnInteractive`** — the full local + server stack, which **depends on** `nninteractive-client`.

Because of that, an editable checkout means installing **both, client first**:

```bash
git clone https://github.com/MIC-DKFZ/nnInteractive
cd nnInteractive

# Recommended: clean slate first, so an older pre-split install can't shadow the
# namespace packages (see the first gotcha below).
pip uninstall -y nnInteractive nninteractive-client

pip install -e ./client   # nninteractive-client (torch-free wire client)
pip install -e .          # nnInteractive (full stack; depends on the client)
```

**Order matters:** installing the editable client *first* satisfies the full package's
dependency from your working tree, so `pip install -e .` will not try to download
`nninteractive-client` from PyPI. If you only develop the remote client, `pip install -e ./client`
on its own is enough (and stays torch-free).

> [!IMPORTANT]
> Two consequences of the namespace-package split that can bite during development:
> - **Do not leave an old, pre-split `nnInteractive` installed.** A monolithic install ships a
>   real `nnInteractive/__init__.py`, which makes `nnInteractive` a *regular* package and
>   **shadows** the editable namespace portions — `import nnInteractive.inference.remote` then
>   fails even though your editable install succeeded. The `pip uninstall` above avoids this.
> - **Editable installs only take effect at interpreter startup.** They work via an import
>   finder registered in a `.pth` file that Python reads when it starts, *not* when you run
>   `pip`. After an editable (re)install, **restart any already-running interpreter** (a Slicer
>   Python console, a Jupyter kernel, …) — otherwise it won't see the package, and
>   `importlib.invalidate_caches()` will not help.

## Getting Started
Here is a minimalistic script that covers the core functionality of nnInteractive:

```python
import os
import torch
import SimpleITK as sitk

# --- Download / locate the trained model weights (~400MB) ---
# License reminder: The official nnInteractive checkpoint is licensed under
# Creative Commons Attribution Non Commercial Share Alike 4.0 (CC BY-NC-SA 4.0).
# See the License section of this readme!
#
# nnInteractive ships a small model-management API. It fetches the list of available
# models from Hugging Face (MIC-DKFZ/nnInteractive), downloads only the selected one,
# reuses it on later runs, and works offline once a model has been downloaded. Models
# are stored under $NNINTERACTIVE_MODEL_DIR (default: ~/.nninteractive).
from nnInteractive.model_management import ensure_model_available, get_default_model_id

# Use the recommended model — i.e. the manifest's default. Don't hard-code a version here, so you
# automatically track whatever is currently recommended:
model_id = get_default_model_id()              # resolves to the recommended id, e.g. "nnInteractive_v1.0"
# To pin a specific model instead, set its id by name (see the ids from
# `nninteractive-available-models` or nnInteractive.model_management.list_models()):
#   model_id = "nnInteractive_v1.0"

# Resolve the id to a local folder: downloads on first use, reuses it afterwards (offline-friendly).
model_path = ensure_model_available(model_id)
# ...later passed to the session via session.initialize_from_trained_model_folder(str(model_path)).

# --- Initialize Inference Session ---
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

session = nnInteractiveInferenceSession(
    device=torch.device("cuda:0"),  # Set inference device
    use_torch_compile=False,  # Compiles the net for faster predictions. The one-time (slow) compile
                              # is paid during initialize_from_trained_model_folder() below (it warms
                              # up automatically), NOT on your first prompt. Worth it for long-lived
                              # processes (the server enables it by default) or longer sessions.
    verbose=False,
    torch_n_threads=os.cpu_count(),  # Use available CPU cores, cap this if your system has a gigantic CPU (64+ cores)
    do_autozoom=True,  # Enables AutoZoom for better patching
)

# Load the trained model
session.initialize_from_trained_model_folder(str(model_path))

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
# Interactions can be freely chained and mixed in any order. Each interaction refines the
# segmentation, and the model writes the updated mask straight into your target buffer
# (in place) after every interaction — you never have to fetch anything back.
#
# Normally you just read the target buffer directly (see "Retrieve Results" below). The only
# reason to look at the return value is if you must propagate the update into your OWN structure
# (a GUI's label layer, a separate array, ...) and want to avoid copying the whole volume every
# time: each add_*_interaction(..., run_prediction=True) RETURNS the bounding box of the region it
# changed, as [[x1, x2], [y1, y2], [z1, z2]] in target-buffer coordinates (None if nothing changed
# / run_prediction=False), so you can copy just that sub-volume.

# Example: Add a **positive** point interaction
# POINT_COORDINATES should be a tuple (x, y, z) specifying the point location.
changed_bbox = session.add_point_interaction(POINT_COORDINATES, include_interaction=True)
# Only needed if you mirror the result elsewhere (otherwise ignore it and read target_tensor):
# if changed_bbox is not None:
#     (x1, x2), (y1, y2), (z1, z2) = changed_bbox
#     my_label_volume[x1:x2, y1:y2, z1:z2] = target_tensor[x1:x2, y1:y2, z1:z2]

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
# - Background must be 0, and scribble must be 1.
# - Use session.preferred_scribble_thickness for optimal results.
#
# ✅ RECOMMENDED (v2): pass a small 2D crop plus its location.
# Scribbles live on a single axis-aligned slice, so one of the three bbox
# dimensions is always size 1 and the in-plane extent typically covers only
# a small region. The cropped array is ORDERS OF MAGNITUDE
# smaller than a full-volume mask for typical annotations, which makes this
# path dramatically faster. Please prefer this
# form in new integrations.
#
# SCRIBBLE_CROP.shape must equal the bbox size. INTERACTION_BBOX uses
# half-open intervals [[x1,x2],[y1,y2],[z1,z2]] in original-image coordinates.
# Example: a scribble drawn on axial slice z=64, covering x∈[100,140), y∈[80,150):
#   SCRIBBLE_CROP    = <ndarray of shape (40, 70, 1), values 0 or 1>
#   INTERACTION_BBOX = [[100, 140], [80, 150], [64, 65]]
session.add_scribble_interaction(
    SCRIBBLE_CROP,
    include_interaction=True,
    interaction_bbox=INTERACTION_BBOX,
)

# Legacy form (still supported, but discouraged): a 3D array matching the
# full original image shape with the scribble baked into one slice.
# session.add_scribble_interaction(SCRIBBLE_IMAGE, include_interaction=True)

# Example: Add a lasso interaction
# - Like scribble but the single slice contains a **closed contour** for the selection.
# - Same recommendation applies: pass a 2D crop + interaction_bbox for a large speedup.
session.add_lasso_interaction(
    LASSO_CROP,
    include_interaction=True,
    interaction_bbox=INTERACTION_BBOX,
)
# Legacy full-volume form (discouraged):
# session.add_lasso_interaction(LASSO_IMAGE, include_interaction=True)

# You can combine any number of interactions as needed. 
# The model refines the segmentation result incrementally with each new interaction.

# --- Retrieve Results ---
# The result already lives in your target buffer: target_tensor IS session.target_buffer (the
# same object, written in place). So just read it — no copy needed to inspect it or save it:
result_np = target_tensor.cpu().numpy()
# sitk.WriteImage(sitk.GetImageFromArray(result_np), "segmentation.nii.gz")

# You only need a COPY if you want to keep this result in memory while you reuse the buffer for
# the next object, because reset_interactions() / reusing the buffer overwrites it in place:
# saved = target_tensor.clone()        # torch  (numpy buffer: target_tensor.copy())

# --- Start a New Object Segmentation ---
session.reset_interactions()  # Clears the target buffer and resets interactions
# (Alternatively, give each object its own fresh buffer instead of resetting:)
# session.set_target_buffer(torch.zeros(img.shape[1:], dtype=torch.uint8))

# Now you can start segmenting the next object in the image.

# --- Set a New Image ---
# Setting a new image also requires setting a new matching target buffer
session.set_image(NEW_IMAGE)
session.set_target_buffer(torch.zeros(NEW_IMAGE.shape[1:], dtype=torch.uint8))

# Enjoy!
```

## Running inference on a remote GPU (client / server)

If the machine running your GUI does not have a powerful GPU, you can run the
model on a remote box and drive it over HTTP with
**`nnInteractiveRemoteInferenceSession`** — a drop-in replacement with the same
public API as the local session.

> [!IMPORTANT]
> **The client needs a server to talk to.** `nnInteractiveRemoteInferenceSession` does nothing
> on its own — it requires an `nninteractive-server` **already running on a (GPU) machine you can
> reach over the network**. Constructing the session connects immediately and raises if the
> server is unreachable. The lightweight `pip install nninteractive-client` only makes sense in
> this setup; if you have no server to connect to, use the local session shown above instead.

The server loads the model once at startup and hosts multiple concurrent client sessions; each
client keeps its own image, target buffer, and interaction state.

Start the server on the GPU box (it downloads the model by name on first use; see
[`SERVER_CLIENT.md`](SERVER_CLIENT.md) for listing/downloading models and where they are stored):

```bash
nninteractive-server \
    --model nnInteractive_v1.0 \
    --host 0.0.0.0 --port 1527 \
    --api-key "$(openssl rand -hex 32)"
```

Then, in the client code, swap the local session for the remote one. No model download or
`initialize_from_trained_model_folder()` is needed — the server already loaded the model:

```python
from nnInteractive.inference.remote import nnInteractiveRemoteInferenceSession

# Requires the server above to be running and reachable at this URL.
session = nnInteractiveRemoteInferenceSession(
    server_url="http://gpu-box.lab:1527",
    api_key="…",
)
# From here on, the API is identical to nnInteractiveInferenceSession
# (set_image / set_target_buffer / add_*_interaction / ...).
```

For full details — installation, authentication, single-user SSH-tunnel setup,
multi-user deployment behind a reverse proxy, concurrency/session model, idle
expiry and heartbeats, GUI integration notes, and troubleshooting — see
[`SERVER_CLIENT.md`](SERVER_CLIENT.md).

### Run the server in Docker

Prefer not to install anything? The server is also published as a Docker image
on the GitHub Container Registry. The default image has the model **baked in**,
so it runs with a single command (a GPU host with the NVIDIA Container Toolkit
is required):

```bash
docker run --gpus all -p 1527:1527 \
    -e NN_INTERACTIVE_API_KEY="$(openssl rand -hex 32)" \
    ghcr.io/mic-dkfz/nninteractive-server:latest
```

A `lite` tag is also available if you'd rather mount your own checkpoint folder
at `/model` instead of using the baked-in weights. For both flavours,
configuration options, and how to build the image yourself, see
[`DOCKER.md`](nnInteractive/inference/server/DOCKER.md).

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
Note that while this repository is available under Apache-2.0 license (see [LICENSE](./LICENSE)), the [model checkpoint](https://huggingface.co/MIC-DKFZ/nnInteractive) is `Creative Commons Attribution Non Commercial Share Alike 4.0`! 

Release model folders ship their own `LICENSE` file whose **first line is the license identifier** (e.g. `CC BY-NC-SA 4.0`); any following lines (such as a link to the full license) are ignored by the tool. At load time this first line is read and exposed as `session.license` so applications can display the model's license prominently. If a checkpoint folder has no `LICENSE` file, the official v1 checkpoint is assumed to be `CC BY-NC-SA 4.0` and any other checkpoint reports `!!MISSING!!`.

# Changelog

<details>
<summary>Click to expand the full version history</summary>

### 2.5.0 - 2026-06-26

- **Lightweight, torch-free `nninteractive-client` package**: the remote client now ships as its own distribution so GUIs / thin clients can drive a remote server without pulling in torch or nnU-Net. It shares the `nnInteractive` import namespace with the full package; the full package depends on it and pins it in lockstep.
- **Model discovery / download by id**: official models are enumerated by a Hugging Face manifest (`MIC-DKFZ/nnInteractive`) and downloaded on demand into `$NNINTERACTIVE_MODEL_DIR` (default `~/.nninteractive`). New `nninteractive-available-models` / `nninteractive-download-model` CLI entry points, and the server gained `--model <id>` to serve an official model by name (`--model-dir` is no longer required). Default `--max-sessions` raised from 1 to 3.
- Predictions and every `add_*_interaction()` call now return the bounding box of the changed region (clipped to the target buffer), so clients that cannot share the buffer can copy only the changed sub-volume.
- Added a **Docker** build for the server.

### 2.4.2 - 2026-06-23

- Reuse the interactions buffer and pinned-memory buffers for the tensor prompt backend (one fewer memcpy); lowered the blosc2 auto-threshold.
- Fixed a crash when an image prompt lies entirely outside the cropped region.
- Disable `torch.compile` on CPU.

### 2.4.0 - 2026-06-19

- Added **undo** support and clearer error messages.
- Reuse pinned-memory buffers for faster prompt compression; fixed a missing typesize in blosc2 compression.
- Disable `torch.compile` on Windows (with a warning).
- Fall back to locating the trainer class in the nnU-Net repo when it is not bundled here.

### 2.3.3 - 2026-06-12

- Choose between torch- and blosc2-backed prompt storage; `auto` (the default) uses a tensor for images smaller than 512×512×1024 and blosc2 otherwise. The blosc2 path is faster thanks to a preallocated output buffer.
- Heartbeats continue during large image transfers.

### 2.3.1 - 2026-06-03

- The client explicitly sets the number of worker threads.

### 2.3.0 - 2026-06-03

- Expose the model license and print it on init; run the `torch.compile` warmup at init.
- Validate remote target-buffer requests.
- Improved blosc2 compression behavior.
- Added the PyPI publishing workflow.

### 2.2.0 - 2026-05-29

- **`torch.compile` support**, enabled by default for the server and triggered at startup so clients never pay the first-prediction compile cost.
- The server gained two independent timeouts: a heartbeat liveness check and an inactivity timeout.

### 2.1.0 - 2026-05-28

- **Remote inference server / client** (`nnInteractiveRemoteInferenceSession`): drive nnInteractive over HTTP from a machine without a GPU. One server hosts multiple concurrent client sessions.
- Large images are chunked during client/server transfer; serialization fixes.
- Simpler install; fixed a bounding box that could extend beyond the image.

### 2.0.0 - 2026-05-28

- Major inference rework — restructured, more readable code.
- Compressed (half-precision) interactions are now the only mode; preprocessing runs on the GPU; substantially reduced VRAM for AutoZoom and refinement.
- New capability / `inference_info.json` format, with bbox sanity and capability checks.
- Removed the pseudo-lasso path; reformatted the codebase with black.

### 1.1.5 - 2026-04-10

- Compatibility with nnU-Net v2.7.0.

### 1.1.4 - 2026-02-19

- Mask refinement now always covers the entire mask.
- Added model-license reminders.

### 1.1.3 - 2025-11-28

- Pin `torch < 2.9.0` to avoid a 3D-convolution memory regression.
- Documentation clarifications (no manual preprocessing in the minimal example).

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

</details>

## Acknowledgments

<p align="left">
  <img src="imgs/Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="imgs/Logos/DKFZ_Logo.png" width="500">
</p>

This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the 
[Division of Medical Image Computing](https://www.dkfz.de/en/medical-image-computing) at DKFZ.
