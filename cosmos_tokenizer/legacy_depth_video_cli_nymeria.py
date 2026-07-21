# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A CLI to run CausalVideoTokenizer on plain videos based on torch.jit.

Usage:
    python3 -m cosmos_tokenizer.video_cli \
        --video_pattern 'path/to/video/samples/*.mp4' \
        --output_dir ./reconstructions \
        --checkpoint_enc ./pretrained_ckpts/CosmosCV_f4x8x8/encoder.jit \
        --checkpoint_dec ./pretrained_ckpts/CosmosCV_f4x8x8/decoder.jit

    Optionally, you can run the model in pure PyTorch mode:
    python3 -m cosmos_tokenizer.video_cli \
        --video_pattern 'path/to/video/samples/*.mp4' \
        --mode=torch \
        --tokenizer_type=CV \
        --temporal_compression=4 \
        --spatial_compression=8 \
        --checkpoint_enc ./pretrained_ckpts/CosmosCV_f4x8x8/encoder.jit \
        --checkpoint_dec ./pretrained_ckpts/CosmosCV_f4x8x8/decoder.jit
"""

# python3 -m cosmos_tokenizer.legacy_depth_video_cli_nymeria --mode=torch --tokenizer_type=DV --temporal_compression=4 --spatial_compression=8 --checkpoint_enc ./Cosmos/checkpoints/Cosmos-0.1-Tokenizer-DV4x8x8/encoder.jit --checkpoint_dec ./Cosmos/checkpoints/Cosmos-0.1-Tokenizer-DV4x8x8/decoder.jit --output_dir ./example_data/nymeria/depth --video_pattern ./example_data/nymeria/depth_npz/202306

import os
from argparse import ArgumentParser, Namespace
from typing import Any
import sys
import tarfile
from io import BytesIO
import numpy as np
from loguru import logger as logging
from cosmos_tokenizer.networks import TokenizerConfigs
from cosmos_tokenizer.utils import (
    get_filepaths,
    get_output_filepath,
    read_video,
    resize_video,
    write_video,
)
import tempfile
import mediapy as mp
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
import tqdm
import glob
import cv2
import subprocess
from multiprocessing import Pool
import multiprocessing
import torch
from torchvision import transforms
from tqdm import tqdm

def _parse_args() -> tuple[Namespace, dict[str, Any]]:
    parser = ArgumentParser(description="A CLI for CausalVideoTokenizer.")
    parser.add_argument(
        "--video_pattern",
        type=str,
        default="path/to/videos/*.mp4",
        help="Glob pattern.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="JIT full Autoencoder model filepath.",
    )
    parser.add_argument(
        "--checkpoint_enc",
        type=str,
        default='pretrained_ckpts/Cosmos-Tokenizer-DV8x16x16/encoder.jit',
        help="JIT Encoder model filepath.",
    )
    parser.add_argument(
        "--checkpoint_dec",
        type=str,
        default=None,
        help="JIT Decoder model filepath.",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["CV", "DV"],
        help="Specifies the tokenizer type.",
    )
    parser.add_argument(
        "--spatial_compression",
        type=int,
        choices=[8, 16],
        default=16,
        help="The spatial compression factor.",
    )
    parser.add_argument(
        "--temporal_compression",
        type=int,
        choices=[4, 8],
        default=8,
        help="The temporal compression factor.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["torch", "jit"],
        default="jit",
        help="Specify the backend: native 'torch' or 'jit' (default: 'jit')",
    )
    parser.add_argument(
        "--short_size",
        type=int,
        default=256,
        help="The size to resample inputs. None, by default.",
    )
    parser.add_argument(
        "--temporal_window",
        type=int,
        default=17,
        help="The temporal window to operate at a time.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Sets the precision, default bfloat16.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for invoking the model.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory."
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=8.0,
        help="Output frames-per-second (FPS).",
    )
    parser.add_argument(
        "--save_input",
        action="store_true",
        help="If on, the input video will be be outputted too.",
    )

    args = parser.parse_args()
    return args


def work(files, savepath, args, gpu_id) -> None:
    """Invokes JIT-compiled CausalVideoTokenizer on an input video."""
    print(f"Processing {len(files)} files in {savepath} ...")
    print(f"Using GPU {gpu_id} ...")
    print(f"savepath: {savepath}")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    if (
        args.checkpoint_enc is None
        and args.checkpoint_dec is None
        and args.checkpoint is None
    ):
        logging.warning(
            "Aborting. Both encoder or decoder JIT required. Or provide the full autoencoder JIT model."
        )
        return

    if args.mode == "torch":
        tokenizer_config = TokenizerConfigs[args.tokenizer_type].value
        tokenizer_config.update(dict(spatial_compression=args.spatial_compression))
        tokenizer_config.update(dict(temporal_compression=args.temporal_compression))
    else:
        tokenizer_config = None

    autoencoder = CausalVideoTokenizer(
        checkpoint=args.checkpoint,
        checkpoint_enc=args.checkpoint_enc,
        checkpoint_dec=args.checkpoint_dec,
        tokenizer_config=tokenizer_config,
        device=args.device,
        dtype=args.dtype,
    )

    if os.path.exists(savepath) is False:
        os.makedirs(savepath)

    for file in files:
        if file.endswith('.npz'):
            if "000017_20230919_s0_andrew_taylor_act2_8ndt6i" not in file:
                continue
            print(f"Processing {file} ...")

            try:
                depth_pred = np.load(file)['arr_0']
                min_depth = depth_pred.min()
                max_depth = depth_pred.max()
                depth_norm = ((depth_pred - min_depth) / (max_depth - min_depth)).clip(0, 1)
                grayscale_video = (depth_norm * 255).astype(np.uint8)
                grayscale_3channel_video = np.stack((grayscale_video,)*3, axis=-1)

                file_name = os.path.basename(file)

                # write grayscale_3channel_video to a temporary mp4 file

                video = grayscale_3channel_video.reshape(1, 16, 256, 256, 3)
                tokens = autoencoder(video, temporal_window=args.temporal_window)
                save_path = os.path.join(savepath, file_name)
                np.savez_compressed(save_path, tokens[0])
            except Exception as e:
                print(f"Error processing {file}: {e}")
                print(file)
                continue

    print(f"Finishing processing {len(files)}.")



if __name__ == "__main__":
    args = _parse_args()
    root_path = args.video_pattern

    folders = os.listdir(root_path)
    folders.sort()


    chunks = [folders[i::4] for i in range(4)]
    for i in range(4):
        print(f"GPU {i} will process {len(chunks[i])} folders.")

    total_len = sum([len(chunk) for chunk in chunks])
    assert total_len == len(folders)
    print(f"Total folders: {total_len}")
    print("#####################Starting multiprocessing... #######################")

    processes = []
    for gpu_id in range(4):
        folder_paths = [os.path.join(root_path, folder) for folder in chunks[gpu_id]]
        p = multiprocessing.Process(target=work, args=(folder_paths, args.output_dir, args, gpu_id))
        processes.append(p)
        p.start()
    
    # # Wait for all processes to complete


