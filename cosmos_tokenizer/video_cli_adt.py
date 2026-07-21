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

# python3 -m cosmos_tokenizer.video_cli_adt --mode=torch --tokenizer_type=DV --temporal_compression=4 --spatial_compression=8 --checkpoint_enc ./Cosmos/checkpoints/Cosmos-0.1-Tokenizer-DV4x8x8/encoder.jit --checkpoint_dec ./Cosmos/checkpoints/Cosmos-0.1-Tokenizer-DV4x8x8/decoder.jit --output_dir ./example_data/token_rgb/ --video_pattern ./example_data/video_256/
# python3 -m cosmos_tokenizer.video_cli_adt --mode=torch --tokenizer_type=DV --temporal_compression=4 --spatial_compression=8 --checkpoint_enc ./Cosmos/checkpoints/Cosmos-0.1-Tokenizer-DV4x8x8/encoder.jit --checkpoint_dec ./Cosmos/checkpoints/Cosmos-0.1-Tokenizer-DV4x8x8/decoder.jit --output_dir ./example_data/token_rgb --video_pattern ./example_data/video_256

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


def _run_eval(filepath, savepath, args, gpu_id) -> None:
    """Invokes JIT-compiled CausalVideoTokenizer on an input video."""

    print(f"Processing folder {filepath} ...")
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

    seq_name = filepath.split('/')[-1]

    files = os.listdir(filepath)
    files.sort()
    for file in files:
        if file.endswith('.mp4'):
            file_path = os.path.join(filepath, file)



            video = mp.read_video(file_path).reshape(1, 16, 256, 256, 3)
            tokens_1 = autoencoder(video[:1], temporal_window=args.temporal_window)
            num_id = file.split('.')[0]
            save_file_path_1 = os.path.join(savepath, num_id + '_' + seq_name + '.npz')
            np.savez_compressed(save_file_path_1, tokens_1[0])



    print(f"Finishing processing {len(files)} in {filepath}.")




def worker(gpu_id, file_paths, args):
    for file_path in file_paths:
        _run_eval(file_path, args.output_dir, args, gpu_id)
    print(f"GPU {gpu_id} finished processing {len(file_paths)} folders.")


if __name__ == "__main__":
    args = _parse_args()
    root_path = args.video_pattern
    folders = os.listdir(root_path)
    folders.sort()

    
    chunks = [folders[i::4] for i in range(4)]

    for i in range(4):
        print(f"GPU {i} will process {len(chunks[i])} folders.")
    processes = []
    for gpu_id in range(4):
        folder_paths = [os.path.join(root_path, folder) for folder in chunks[gpu_id]]
        p = multiprocessing.Process(target=worker, args=(gpu_id, folder_paths, args))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()


