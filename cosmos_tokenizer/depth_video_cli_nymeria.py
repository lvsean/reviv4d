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
"""Encode depth npz files with Cosmos discrete video tokenizer."""

import os
import multiprocessing
from argparse import ArgumentParser, Namespace

import cv2
import numpy as np
from loguru import logger as logging

from cosmos_tokenizer.networks import TokenizerConfigs
from cosmos_tokenizer.video_lib import CausalVideoTokenizer


# python3 -m cosmos_tokenizer.depth_video_cli_nymeria --mode=torch --tokenizer_type=DV --temporal_compression=8 --spatial_compression=16 --temporal_window=32 --checkpoint_enc ./Cosmos/checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/encoder.jit --checkpoint_dec ./Cosmos/checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/decoder.jit --video_pattern ./example_data/nymeria/depth_metric_npz_512_vda --output_dir ./example_data/nymeria/depth_tokens_vda --recursive

def _parse_args() -> Namespace:
    parser = ArgumentParser(description="Encode depth files with CausalVideoTokenizer.")
    parser.add_argument(
        "--video_pattern",
        type=str,
        required=True,
        help="Input directory containing depth npz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for tokenized npz files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional full Autoencoder model filepath.",
    )
    parser.add_argument(
        "--checkpoint_enc",
        type=str,
        default="./Cosmos/checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/encoder.jit",
        help="Encoder checkpoint filepath.",
    )
    parser.add_argument(
        "--checkpoint_dec",
        type=str,
        default="./Cosmos/checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/decoder.jit",
        help="Decoder checkpoint filepath.",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["CV", "DV"],
        default="DV",
        help="Tokenizer type.",
    )
    parser.add_argument(
        "--spatial_compression",
        type=int,
        choices=[8, 16],
        default=16,
        help="Spatial compression factor.",
    )
    parser.add_argument(
        "--temporal_compression",
        type=int,
        choices=[4, 8],
        default=8,
        help="Temporal compression factor.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["torch", "jit"],
        default="torch",
        help="Backend mode.",
    )
    parser.add_argument(
        "--temporal_window",
        type=int,
        default=32,
        help="Temporal window size for tokenizer inference.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Inference precision.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for tokenizer inference.",
    )
    parser.add_argument(
        "--short_size",
        type=int,
        default=None,
        help="Optional square resize before tokenization.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for npz files under --video_pattern.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=4,
        help="Number of worker processes / GPUs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def _list_depth_files(root_path: str, recursive: bool) -> list[str]:
    if os.path.isfile(root_path):
        return [root_path] if root_path.endswith(".npz") else []

    if recursive:
        depth_files = []
        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.endswith(".npz"):
                    depth_files.append(os.path.join(dirpath, filename))
        return sorted(depth_files)

    return sorted(
        os.path.join(root_path, filename)
        for filename in os.listdir(root_path)
        if filename.endswith(".npz")
    )


def _load_depth_array(filepath: str) -> np.ndarray:
    with np.load(filepath, allow_pickle=True) as data:
        if "arr_0" in data:
            depth = data["arr_0"]
        elif len(data.files) == 1:
            depth = data[data.files[0]]
        else:
            raise ValueError(f"Unable to infer depth array key from {filepath}: {data.files}")

    depth = np.asarray(depth).squeeze()
    if depth.ndim == 2:
        depth = depth[np.newaxis, ...]
    elif depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    if depth.ndim != 3:
        raise ValueError(f"Expected depth array with shape [T,H,W], got {depth.shape}")
    return depth.astype(np.float32)


def _resize_video(video: np.ndarray, short_size: int | None) -> np.ndarray:
    if short_size is None:
        return video

    resized = [
        cv2.resize(frame, (short_size, short_size), interpolation=cv2.INTER_LINEAR)
        for frame in video
    ]
    return np.stack(resized, axis=0)


def _depth_to_tokenizer_video(depth: np.ndarray, short_size: int | None) -> np.ndarray:
    depth = _resize_video(depth, short_size)
    finite_mask = np.isfinite(depth)
    if not finite_mask.any():
        gray = np.zeros(depth.shape, dtype=np.uint8)
    else:

        min_depth = 0.0
        max_depth = 15.0
        if max_depth <= min_depth:
            gray = np.zeros(depth.shape, dtype=np.uint8)
        else:
            normalized = np.zeros_like(depth, dtype=np.float32)
            normalized[finite_mask] = (depth[finite_mask] - min_depth) / (max_depth - min_depth)
            gray = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)

    return np.repeat(gray[..., np.newaxis], 3, axis=-1)


def _build_tokenizer(args: Namespace) -> CausalVideoTokenizer:
    if (
        args.checkpoint_enc is None
        and args.checkpoint_dec is None
        and args.checkpoint is None
    ):
        raise ValueError("Provide encoder/decoder checkpoints or a full checkpoint.")

    if args.mode == "torch":
        tokenizer_config = dict(TokenizerConfigs[args.tokenizer_type].value)
        tokenizer_config.update(dict(spatial_compression=args.spatial_compression))
        tokenizer_config.update(dict(temporal_compression=args.temporal_compression))
    else:
        tokenizer_config = None

    return CausalVideoTokenizer(
        checkpoint=args.checkpoint,
        checkpoint_enc=args.checkpoint_enc,
        checkpoint_dec=args.checkpoint_dec,
        tokenizer_config=tokenizer_config,
        device=args.device,
        dtype=args.dtype,
    )


def _worker(gpu_id: int, filepaths: list[str], args: Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tokenizer = _build_tokenizer(args)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"GPU {gpu_id} processing {len(filepaths)} files.")
    for filepath in filepaths:
        if os.path.isfile(args.video_pattern):
            relative_path = os.path.basename(filepath)
        else:
            relative_path = os.path.relpath(filepath, args.video_pattern)
        save_path = os.path.join(args.output_dir, relative_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path) and not args.overwrite:
            continue

        try:
            depth = _load_depth_array(filepath)
            video = _depth_to_tokenizer_video(depth, args.short_size)[np.newaxis, ...]
            print(f"Tokenizing {filepath} with shape {video.shape} ...")
            tokens = tokenizer(video, temporal_window=args.temporal_window)
            print(f"Saving tokens to {save_path} with shape {tokens.shape} ...")
            np.savez_compressed(save_path, tokens[0])
        except Exception as exc:
            logging.exception(f"Failed to process {filepath}: {exc}")

    print(f"GPU {gpu_id} finished.")


if __name__ == "__main__":
    args = _parse_args()
    filepaths = _list_depth_files(args.video_pattern, recursive=args.recursive)
    if not filepaths:
        raise FileNotFoundError(f"No npz files found under {args.video_pattern}")

    num_workers = max(1, min(args.num_gpus, len(filepaths)))
    chunks = [filepaths[i::num_workers] for i in range(num_workers)]
    for gpu_id, chunk in enumerate(chunks):
        print(f"GPU {gpu_id} will process {len(chunk)} files.")

    processes = []
    for gpu_id, chunk in enumerate(chunks):
        process = multiprocessing.Process(target=_worker, args=(gpu_id, chunk, args))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
