import os
import yaml
import argparse
from typing import List, Tuple
import numpy as np
from supervoxel import SuperVoxelGenerator
from metadata import generate_fg_locations
from tqdm import tqdm
import torch
import gc


def list_and_clear_tensors_on_gpu():

    gpu_tensors = []  # Collect GPU tensors to analyze them
    tensor_sizes = []  # Collect tensor sizes to analyze them

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                print(f"Tensor: {type(obj)}, size: {obj.size()}, device: {obj.device}")
                gpu_tensors.append(obj)
                tensor_sizes.append(obj.nbytes)
        except Exception as e:
            pass  # Handle any inspection errors

    # Delete all tensors found
    print(f"Found {len(gpu_tensors)} tensors on GPU.")
    bytes_sum = sum(tensor_sizes)
    print(f"Total size: {bytes_sum / 1024 ** 3} GB")
    for tensor in gpu_tensors:
        del tensor  # Remove local references to the tensors

    # Trigger garbage collection
    gc.collect()

    # Empty CUDA cache
    torch.cuda.empty_cache()

    print("Cleared GPU tensors and memory.")


def run(input_folder: str, output_folder: str, config: str):
    """
    Run the SuperVoxel generation using SAM

    :param input_folder: Path to folder containing the files to process
    :param output_folder: Path to output folder. If not provided, the output will be saved in the same folder as the dataset.
    """
    if output_folder is None:
        output_folder = os.path.join(input_folder, os.pardir, "supervoxel")
    os.makedirs(output_folder, exist_ok=True)

    # Load congiguration file
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    gen = SuperVoxelGenerator(input_folder, output_folder, config)

    # List of files
    list_of_files = [
        f for f in os.listdir(input_folder) if f.endswith(config["file_format"]) and config["excluded_strings"] not in f
    ]

    # Shuffle the list of files
    np.random.shuffle(list_of_files)
    for image_name in tqdm(list_of_files):
        out_path = os.path.join(output_folder, image_name)
        if os.path.exists(out_path):
            continue

        image_data, metadata = gen.reader_writer.read(os.path.join(input_folder, image_name))

        if len(image_data.shape) == 4:
            image_data = image_data[0]
            # chatch OOM error
        try:
            out_img = None
            out_img = gen.sam_supervoxel(image_data)
        except torch.OutOfMemoryError:
            print("OOM error for image:", image_name)
            if out_img is not None:
                del out_img
            if gen is not None:
                del gen
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gen = SuperVoxelGenerator(input_folder, output_folder, config)
            continue

        if os.path.exists(out_path):
            continue
        gen.reader_writer.write(out_img, metadata, out_path)


def run_entrypoint():
    parser = argparse.ArgumentParser(description="Run SuperVoxel generation using SAM")
    parser.add_argument(
        "-i",
        "-input_folder",
        type=str,
        help="Path to folder containing the images. They can be in raw Nifit format \
                         or using the new nnUNet supported bloscv2, depending on the file format provided in the config file.",
    )
    parser.add_argument(
        "-o",
        "-output_folder",
        type=str,
        help="Path to output folder. If not provided, the output will be saved in the \
                         same parent folder as the dataset and named 'supervoxel'.",
    )
    parser.add_argument(
        "-c",
        "-config",
        type=str,
        help="Path to configuration file containing the parameters for the SuperVoxel generation.",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../configs/nnUNet_preprocessed.yaml"),
    )
    args = parser.parse_args()

    run(args.i, args.o, args.c)


def run_save_fg_locations_entrypoint():
    parser = argparse.ArgumentParser(description="Run generation of pkl files for nnUNet")
    parser.add_argument(
        "-supervoxel_folder",
        type=str,
        help="Path to folder containing the supervoxel masks",
        default="/home/m574s/PhD/projects/SuperVoxel/supervoxels/",
    )
    parser.add_argument("-np", "-num_processes", type=int, help="Number of processes to use")
    args = parser.parse_args()

    generate_fg_locations(args.supervoxel_folder, args.np)
