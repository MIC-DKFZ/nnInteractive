import pandas as pd
import numpy as np
import math
import pickle as pkl
import os
from tqdm import tqdm

from reader import BloscReaderWriter
from typing import List, Tuple, Union
import multiprocessing as mp
from functools import partial


def sample_foreground_locations(
    seg: np.ndarray,
    classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
    seed: int = 1234,
    verbose: bool = False,
):
    num_samples = 100
    # sparse
    rndst = np.random.RandomState(seed)
    class_locs = {}
    foreground_mask = seg != 0
    foreground_coords = np.argwhere(foreground_mask)
    seg = seg[foreground_mask]
    del foreground_mask
    unique_labels = pd.unique(seg.ravel())

    # We don't need more than 1e7 foreground samples. That's insanity. Cap here
    if len(foreground_coords) > 1e7:
        take_every = math.floor(len(foreground_coords) / 1e7)
        # keep computation time reasonable
        if verbose:
            print(f"Subsampling foreground pixels 1:{take_every} for computational reasons")
        foreground_coords = foreground_coords[::take_every]
        seg = seg[::take_every]

    for c in classes_or_regions:
        k = c if not isinstance(c, list) else tuple(c)

        # check if any of the labels are in seg, if not skip c
        if isinstance(c, (tuple, list)):
            if not any([ci in unique_labels for ci in c]):
                class_locs[k] = []
                continue
        else:
            if c not in unique_labels:
                class_locs[k] = []
                continue

        if isinstance(c, (tuple, list)):
            mask = seg == c[0]
            for cc in c[1:]:
                mask = mask | (seg == cc)
            all_locs = foreground_coords[mask]
        else:
            mask = seg == c
            all_locs = foreground_coords[mask]
        if len(all_locs) == 0:
            class_locs[k] = []
            continue
        target_num_samples = min(num_samples, len(all_locs))

        selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
        class_locs[k] = selected
        if verbose:
            print(c, target_num_samples)
        seg = seg[~mask]
        foreground_coords = foreground_coords[~mask]
    return class_locs


def process_file(file, supervoxel_folder, bloscio):
    out_file = os.path.join(supervoxel_folder, file.replace(".b2nd", ".pkl"))

    if os.path.exists(out_file):
        return

    # Load the supervoxel file
    supervoxel_arr, _ = bloscio.read(os.path.join(supervoxel_folder, file))

    assert supervoxel_arr.ndim == 4, "The supervoxel array should have 4 dimensions, failed for file: " + file

    all_class_locs = []
    for submask in supervoxel_arr:
        assert submask.ndim == 3, "The submask should have 3 dimensions, failed for file: " + file
        all_class_locs.append(sample_foreground_locations(submask, [1]))

    # Save the foreground locations
    with open(out_file, "wb") as f:
        pkl.dump(all_class_locs, f)


def generate_fg_locations(supervoxel_folder, num_processes=4):
    """
    Generate the foreground locations for the supervoxels

    Parameters:
        supervoxel_folder (str): The path to the folder containing the supervoxel files
    """
    bloscio = BloscReaderWriter()
    # Load the supervoxel files
    supervoxel_files = [f for f in os.listdir(supervoxel_folder) if f.endswith(".b2nd")]

    if num_processes == 1:
        for file in tqdm(supervoxel_files):
            process_file(file, supervoxel_folder, bloscio)
    else:
        with mp.Pool(num_processes) as pool:
            list(
                tqdm(
                    pool.imap(
                        partial(process_file, supervoxel_folder=supervoxel_folder, bloscio=bloscio), supervoxel_files
                    ),
                    total=len(supervoxel_files),
                )
            )
