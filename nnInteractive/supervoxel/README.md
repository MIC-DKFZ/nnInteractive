# nnInteractive SuperVoxel: Supervoxel Generation with SAM and SAM2 for 3D Medical Imaging

## Overview


`nnInteractive SuperVoxel` is a module within the `nnInteractive` framework for generating supervoxels — high-quality, variable-sized 3D pseudo-labels - for 3D medical images. This is accomplished using the **automatic mask generation** capabilities of [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) and the **mask propagation** in [SAM2](https://github.com/facebookresearch/sam2). This module replaces traditional superpixel-based approaches (e.g., SLIC, Felzenszwalb) with foundation model–driven supervoxels. The generated **pseudo-labels** (foreground masks) can e.g. be used to train promptable segmentation models. 

## 🧠 How It Works

1. **Axial Sampling + Segmentation (SAM):**  
   A slice is selected along the axial axis. SAM segments a set of visible objects with high confidence (IoU ≥ 92%).

2. **3D Mask Propagation (SAM2):**  
   A subset of `n` masks are treated as "keyframes" and passed through SAM2's temporal propagation to build 3D objects across slices. `n` is defined via `masks_per_image` in the `config.yaml`.

3. **Output:**  
   A 4D image with the same spatial shape as the input image (e.g. `[H, W, D]`) and `n` additional channels, where each channel represents a pseudo-label for one segmented object. **Note*:* The actual number of output masks can be lower if not enough high-confidence masks are detected.

---

## ⚙️ Installation

1. Clone this repository (as part of `nnInteractive` or standalone).
2. Install dependencies (first a modified version of SAM2 and then this repo):

```bash
cd src/sam2/
pip install -e .
cd ../..
pip install -e .
```

---

## 📦 Model Checkpoints

You’ll need to download pretrained weights for SAM and SAM2 and specify their paths in the config file.

- SAM 👉 [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- SAM2 👉 [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

Update `configs/config.yaml` with the local paths to these files.

---

## 🚀 Usage

### 🔧 Generate SuperVoxels

```bash
SuperVoxel_generate \
  -i /path/to/input/images \
  -o /path/to/output/folder \
  -c /path/to/config.yaml
```

Output: A 4D image with the same spatial size as the input and one dimension for generated pseudo-label. The number of channels depends on the `masks_per_image` setting and may be fewer if confidence thresholds aren't met.

📌 *Input formats supported:*
- Standard `.nii.gz` (NIfTI) or other formats supported by SimpleITK
- `.bloscv2` format (used by `nnUNetv2_preprocessed`)  
Ensure the config specifies the correct format.

This computation is very compute and VRAM expensive. There are failsave mechanisms in the code for OOM erros but it can still sometimes struggle with large images.

---

## 🗂️ Config Files

Template config files for preprocessed nnUNet datasets and custom paths can be found in the [`configs/`](configs/) directory.

- `nnUNet_preprocessed.yaml` – template for datasets using nnUNetv2 formats
- `config.yaml` – edit this with SAM/SAM2 paths and dataset-specific parameters

---

### Generate nnUNet-Compatible Foreground Prompts

Use this to create `.pkl` files containing foreground locations from the generated supervoxels, which `nnUNet` can use for training.

```bash
SuperVoxel_save_fg_location \
  -supervoxel_folder /path/to/supervoxel/masks \
  -np 4  # number of parallel processes
```

---

## 📜 License

This project is licensed under the **Apache 2.0 License**.

---

## 📬 Contact

For questions, bugs, or collaboration inquiries, feel free to reach out via email 📧 maximilian.rokuss@dkfz-heidelberg.de or GitHub issues.



