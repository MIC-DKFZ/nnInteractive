#!/usr/bin/env python3
"""
CVPR25 – Foundation Models for Interactive 3D Biomedical Image Segmentation
Skeleton inference script.

You only need to replace the `run_inference()` function with your model‑specific
code.  Everything else takes care of
  • reading the input image + prompts,
  • passing the relevant information to your model,
  • saving the prediction in the expected format.

During evaluation the script is called exactly once for every interaction
step (bbox prediction + 5 click refinements).  The evaluator will overwrite
the same input file between calls, injecting updated clicks and the previous
prediction (`prev_pred`).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

from nnunetv2.utilities.helpers import empty_cache


# --------------------------------------------------------------------------- #
#                           ===  EDIT BELOW  ===                              #
# --------------------------------------------------------------------------- #

def run_inference(
    image: np.ndarray,
    spacing: tuple[float, float, float],
    bbox: list[dict] | None,
    clicks: list[dict] | None,
    clicks_order: list[list[str]] | None,
    prev_pred: np.ndarray | None,
) -> np.ndarray:
    """
    Stub performing **one** forward pass of your model.

    Parameters
    ----------
    image : (D, H, W) np.ndarray
        Raw image volume (usually float32).  *No preprocessing applied*.
    spacing : (3,) tuple of float
        Physical voxel spacing (z, y, x) in millimetres.
    bbox : list of dict | None
        Bounding‑box prompt(s).  The dict structure is shown in the challenge
        description; may be absent in refinement iterations.
    clicks : list of dict | None
        Fore‑ and background click dictionaries for every class.
    prev_pred : (D, H, W) np.ndarray | None
        Segmentation from the previous iteration.  May be `None` for the first
        call.

    Returns
    -------
    seg : (D, H, W) np.ndarray, dtype=uint8
        Multi‑class segmentation mask.  Background **must** be 0;
        classes start from 1 … N.  Make sure dtype is `np.uint8`.
    """
    session = nnInteractiveInferenceSession(
        device=torch.device('cuda', 0),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        use_pinned_memory=True
    )
    session.initialize_from_trained_model_folder(
        model_training_output_dir=CHECKPOINT_DIR,
        use_fold='all',
    )
    session.set_image(image[None].astype(np.float32))
    target_buffer = torch.zeros(image.shape, dtype=torch.uint8, device='cpu')
    session.set_target_buffer(target_buffer)
    result = torch.zeros(image.shape, dtype=torch.uint8)
    num_objects = len(bbox) if bbox is not None else len(clicks)
    if bbox is not None and clicks is not None:
        assert len(bbox) == len(clicks), ('Both bboxs and clicks lists are provided but with different length '
                                          'suggesting different number of objects. This is not supported by this script '
                                          'and it was not communicated by the organizing team that such cases exist '
                                          'or how they are supposed to be handled.')
    for oid in range(1, num_objects + 1):
        # place previous segmentation
        if prev_pred is not None:
            session.add_initial_seg_interaction((prev_pred == oid).astype(np.uint8), run_prediction=False)
        else:
            session.reset_interactions()
        if bbox is not None:
            bbox_here = bbox[oid - 1]
            bbox_here = [
                [bbox_here['z_min'], bbox_here['z_max'] + 1],
                [bbox_here['z_mid_y_min'], bbox_here['z_mid_y_max'] + 1],
                [bbox_here['z_mid_x_min'], bbox_here['z_mid_x_max'] + 1]
                ]
            session.add_bbox_interaction(bbox_here, include_interaction=True, run_prediction=False)
        if clicks is not None:
            clicks_here = clicks[oid - 1]
            clicks_order_here = clicks_order[oid - 1]
            fg_ptr = bg_ptr = 0
            for kind in clicks_order_here:
                if kind == 'fg':
                    click = clicks_here['fg'][fg_ptr]
                    fg_ptr += 1
                else:
                    click = clicks_here['bg'][bg_ptr]
                    bg_ptr += 1

                print(f"Class {oid}: {kind} click at {click}")
                session.add_point_interaction(click, include_interaction=kind == 'fg', run_prediction=False)
        # now run inference on the last interaction center
        session.new_interaction_centers = [session.new_interaction_centers[-1]]
        session.new_interaction_zoom_out_factors = [session.new_interaction_zoom_out_factors[-1]]
        session._predict()
        result[session.target_buffer > 0] = oid
    del session
    empty_cache(torch.device('cuda', 0))
    return result.cpu().numpy()

# --------------------------------------------------------------------------- #
#                         ===  DO NOT EDIT BELOW ===                          #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case_path", required=True, help="Path to the input *.npz")
    p.add_argument("--save_path", required=True, help="Path to write output *.npz")
    return p.parse_args()

# adapt this for the docker!!!
CHECKPOINT_DIR = '/home/isensee/drives/checkpoints/nnUNet_results_remake_nnInteractive_aliveCheck/Dataset165_nnInteractiveV4/nnInteractiveTrainer_CVPR2025_Challenge_1000ep__nnUNetResEncUNetLPlans_noResampling__3d_fullres_ps192_bs24'


def main() -> None:
    args = parse_args()
    case_path = Path(args.case_path)
    save_path = Path(args.save_path)

    if not case_path.is_file():
        sys.exit(f"[predict.py] ERROR: {case_path} not found.")

    # ---------------------- Load input & prompts -------------------------- #
    data = np.load(case_path, allow_pickle=True)
    image        = data["imgs"]
    spacing      = tuple(data["spacing"])
    bbox         = data.get("boxes")         # bounding boxes
    clicks       = data.get("clicks")        # fg/bg clicks per class
    clicks_order = data.get("clicks_order")  # order of click types
    prev_pred    = data.get("prev_pred")     # from last iteration

    # --------------------------- Inference -------------------------------- #
    seg = run_inference(image, spacing, bbox, clicks, clicks_order, prev_pred)

    # ------------------------- Save prediction ---------------------------- #
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, segs=seg.astype(np.uint8))
    print(f"[predict.py] Saved prediction to {save_path}")

if __name__ == "__main__":
    main()