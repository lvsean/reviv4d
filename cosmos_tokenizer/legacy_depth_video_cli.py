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
# Data root, override with the REVIV_DATA_ROOT env var (default: ./example_data).
DATA_ROOT = os.environ.get("REVIV_DATA_ROOT", "./example_data")

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
        default=8,
        help="The spatial compression factor.",
    )
    parser.add_argument(
        "--temporal_compression",
        type=int,
        choices=[4, 8],
        default=4,
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


def _run_eval(filepath, args, gpu_id) -> None:
    """Invokes JIT-compiled CausalVideoTokenizer on an input video."""
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

    # logging.info(
    # )
    autoencoder = CausalVideoTokenizer(
        checkpoint=args.checkpoint,
        checkpoint_enc=args.checkpoint_enc,
        checkpoint_dec=args.checkpoint_dec,
        tokenizer_config=tokenizer_config,
        device=args.device,
        dtype=args.dtype,
    )
    
    print(filepath, flush=True)
    with tarfile.open(filepath, "r") as tar:
        tar_out = tarfile.open(get_output_filepath(filepath, output_dir=args.output_dir), 'w')
        for member in tqdm.tqdm(tar.getmembers()):
            if member.isfile() and member.name.endswith('.npz'):
                file_content = tar.extractfile(member).read()
                depth_pred = np.load(BytesIO(file_content))['arr_0']

                min_depth = depth_pred.min()
                max_depth = depth_pred.max()
                depth_norm = ((depth_pred - min_depth) / (max_depth - min_depth)).clip(0, 1)
                grayscale_video = (depth_norm * 255).astype(np.uint8)
                grayscale_3channel_video = np.stack((grayscale_video,)*3, axis=-1)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
                    temp_output_path = temp_output.name

                # downsample video to 8 fps and center crop and resize
                crop_size = 480
                resize_size = 256

                num_frames, input_height, input_width, _ = grayscale_3channel_video.shape
                input_frame_rate = 30  # Ensure this is defined
                output_frame_rate = args.output_fps  # Ensure this is defined

                # Calculate crop dimensions
                x_offset = (input_width - crop_size) // 2
                y_offset = (input_height - crop_size) // 2

                # ffmpeg_command = [
                #     'ffmpeg', 
                #     '-i', temp_input_path,  # Input file
                #     '-c:v', 'libx264',  # Video codec
                #     '-crf', '18',
                #     '-preset', 'slow',
                #     '-y',  # Overwrite output file
                #     temp_output_path
                # ]
                # subprocess.run(
                # )

                ffmpeg_command = [
                    'ffmpeg',
                    '-loglevel', 'quiet',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-pix_fmt', 'rgb24',
                    '-s', f'{input_width}x{input_height}',
                    '-r', str(input_frame_rate),
                    '-i', '-',  # Input from stdin
                    '-vf', f"fps={output_frame_rate},crop={crop_size}:{crop_size}:{x_offset}:{y_offset},scale={resize_size}:{resize_size}:flags=lanczos",
                    '-c:v', 'libx264',
                    '-crf', '18',
                    '-preset', 'slow',
                    '-y',  # Overwrite output file
                    temp_output_path
                ]

                # Start ffmpeg subprocess
                process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

                # Write frames to ffmpeg stdin
                for frame in grayscale_3channel_video:
                    process.stdin.write(frame.tobytes())

                process.stdin.close()
                process.wait()

                with open(temp_output_path, 'rb') as f:
                    downsampled_content = f.read()

                video = mp.decompress_video(downsampled_content)
                batch_video = video[np.newaxis, ...]
                
                output_token = autoencoder(batch_video, temporal_window=args.temporal_window)[0]
                bytes_io1 = BytesIO()
                np.savez_compressed(bytes_io1, output_token)
                bytes_io1.seek(0)
                filename = os.path.basename(member.name)
                if '.mp4' in filename:
                    filename = filename.replace('.mp4', '')
                file_data = bytes_io1.getvalue()
                tarinfo1 = tarfile.TarInfo(name=filename)
                tarinfo1.size = len(file_data)
                # Write to the tarfile
                tar_out.addfile(tarinfo1, BytesIO(file_data))

        tar_out.close()


def check_and_delete_tar(tar_path):
    try:
        # Attempt to open the tar file and list its contents
        with tarfile.open(tar_path, 'r') as tar:
            tar.getmembers()
    except (tarfile.TarError, EOFError, IOError) as e:
        # If an error occurs, the tar file is considered invalid
        print(f"Error: The tar file '{tar_path}' is invalid. Deleting the file.")


if __name__ == "__main__":
    x = []
    x.extend(glob.glob(f'{DATA_ROOT}/datasets_aligned/depth/holoassist/video_tar/shard*.tar'))
    x.extend(glob.glob(f'{DATA_ROOT}/datasets_aligned/depth/holoassist/video_tar/val*.tar'))
    filepaths = sorted(x)
    args = _parse_args()
    gpu_ids = list(range(4))
    with Pool(4) as pool:
        pool.starmap(_run_eval, [(filepath, args, gpu_ids[i % 4]) for i, filepath in enumerate(filepaths)])
    