"""Multi-node depth tokenization for Nymeria depth npz files.

This keeps the Nymeria depth-tokenization flow, but shards the file list across
the global Slurm/torchrun world size so every GPU across all nodes contributes.
"""

import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from loguru import logger as logging

from cosmos_tokenizer.networks import TokenizerConfigs
from cosmos_tokenizer.video_lib import CausalVideoTokenizer

NYMERIA_DEPTH_ALIGNED_SUBDIR = "unidepth_depth_aligned"


def _parse_args():
    parser = ArgumentParser(description="Multi-node Nymeria depth tokenizer.")
    parser.add_argument(
        "--video_pattern",
        type=str,
        required=True,
        help="Input directory containing depth npz files, or a single npz file.",
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
        help="Requested device. Default uses the Slurm-assigned GPU.",
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--depth_min",
        type=float,
        default=0.0,
        help="Global minimum depth for fixed normalization.",
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=15.0,
        help="Global maximum depth for fixed normalization.",
    )
    parser.add_argument(
        "--normalize_per_file",
        action="store_true",
        help="Use per-file min/max depth normalization instead of fixed depth_min/depth_max.",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=None,
        help="Local rank injected by torchrun.",
    )
    return parser.parse_args()


def _parse_env_int(names, default):
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return default


def _resolve_rank_info(local_rank_arg):
    rank = _parse_env_int(("RANK", "SLURM_PROCID"), 0)
    world_size = _parse_env_int(("WORLD_SIZE", "SLURM_NTASKS"), 1)
    local_rank = local_rank_arg
    if local_rank is None:
        local_rank = _parse_env_int(("LOCAL_RANK", "SLURM_LOCALID"), 0)
    return rank, world_size, local_rank


def _select_cuda_device(local_rank):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for encode_depth_nymeria.py.")

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count == 1:
        device_index = 0
    elif 0 <= local_rank < visible_gpu_count:
        device_index = local_rank
    else:
        raise RuntimeError(
            "LOCAL_RANK={} is invalid for {} visible GPU(s).".format(
                local_rank, visible_gpu_count
            )
        )

    torch.cuda.set_device(device_index)
    return "cuda:{}".format(device_index), visible_gpu_count


def _resolve_device(requested_device, local_rank):
    if requested_device == "cuda":
        return _select_cuda_device(local_rank)
    return requested_device, torch.cuda.device_count() if torch.cuda.is_available() else 0


def _build_tokenizer(args, device):
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
        device=device,
        dtype=args.dtype,
    )


def _list_depth_files(root_path, recursive):
    if os.path.isfile(root_path):
        return [root_path] if root_path.endswith(".npz") else []

    if recursive:
        nymeria_depth_files = []
        depth_files = []
        for dirpath, _, filenames in os.walk(root_path):
            if os.path.basename(dirpath) == NYMERIA_DEPTH_ALIGNED_SUBDIR:
                for filename in filenames:
                    if filename.endswith(".npz"):
                        nymeria_depth_files.append(os.path.join(dirpath, filename))
                continue

            for filename in filenames:
                if filename.endswith(".npz"):
                    depth_files.append(os.path.join(dirpath, filename))
        if nymeria_depth_files:
            return sorted(nymeria_depth_files)
        return sorted(depth_files)

    return sorted(
        os.path.join(root_path, filename)
        for filename in os.listdir(root_path)
        if filename.endswith(".npz")
    )


def _load_depth_array(filepath):
    with np.load(filepath, allow_pickle=True) as data:
        if "depth" in data:
            depth = data["depth"]
        elif "arr_0" in data:
            depth = data["arr_0"]
        elif len(data.files) == 1:
            depth = data[data.files[0]]
        else:
            raise ValueError(
                "Unable to infer depth array key from {}: {}".format(
                    filepath, data.files
                )
            )

    depth = np.asarray(depth).squeeze()
    if depth.ndim == 2:
        depth = depth[np.newaxis, ...]
    elif depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    if depth.ndim != 3:
        raise ValueError("Expected depth array with shape [T,H,W], got {}".format(depth.shape))
    return depth.astype(np.float32)


def _resize_video(video, short_size):
    if short_size is None:
        return video

    resized = [
        cv2.resize(frame, (short_size, short_size), interpolation=cv2.INTER_LINEAR)
        for frame in video
    ]
    return np.stack(resized, axis=0)


def _depth_to_tokenizer_video(depth, args):
    depth = _resize_video(depth, args.short_size)
    finite_mask = np.isfinite(depth)
    if not finite_mask.any():
        gray = np.zeros(depth.shape, dtype=np.uint8)
    else:
        if args.normalize_per_file:
            valid_depth = depth[finite_mask]
            min_depth = float(valid_depth.min())
            max_depth = float(valid_depth.max())
        else:
            min_depth = float(args.depth_min)
            max_depth = float(args.depth_max)

        if max_depth <= min_depth:
            raise ValueError(
                "Invalid depth normalization range: min_depth={} max_depth={}".format(
                    min_depth, max_depth
                )
            )

        normalized = np.zeros_like(depth, dtype=np.float32)
        normalized[finite_mask] = (depth[finite_mask] - min_depth) / (max_depth - min_depth)
        gray = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)

    return np.repeat(gray[..., np.newaxis], 3, axis=-1)


def _build_output_path(filepath, input_root, output_dir):
    path_parts = os.path.normpath(filepath).split(os.sep)
    if NYMERIA_DEPTH_ALIGNED_SUBDIR in path_parts:
        return os.path.join(output_dir, os.path.basename(filepath))

    if os.path.isfile(input_root):
        relative_path = os.path.basename(filepath)
    else:
        relative_path = os.path.relpath(filepath, input_root)
    return os.path.join(output_dir, relative_path)


def _has_valid_npz(output_path):
    if not os.path.isfile(output_path):
        return False

    try:
        if os.path.getsize(output_path) == 0:
            return False
        with np.load(output_path, allow_pickle=False) as data:
            return len(data.files) > 0
    except Exception:
        return False


def _collect_pending_files(filepaths, args):
    pending = []
    ready = 0

    for filepath in filepaths:
        save_path = _build_output_path(filepath, args.video_pattern, args.output_dir)
        if not args.overwrite and _has_valid_npz(save_path):
            ready += 1
            continue
        pending.append((filepath, save_path))

    return pending, ready


def _save_npz(output_path, array):
    tmp_output_path = "{}.tmp.npz".format(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        if os.path.exists(tmp_output_path):
            os.remove(tmp_output_path)
        np.savez_compressed(tmp_output_path, array)
        os.replace(tmp_output_path, output_path)
    except Exception:
        if os.path.exists(tmp_output_path):
            os.remove(tmp_output_path)
        raise


def _process_files(tokenizer, pending_files, args, rank):
    processed = 0
    failed = 0

    os.makedirs(args.output_dir, exist_ok=True)
    for filepath, save_path in pending_files:
        try:
            depth = _load_depth_array(filepath)
            video = _depth_to_tokenizer_video(depth, args)[np.newaxis, ...]
            with torch.inference_mode():
                tokens = tokenizer(video, temporal_window=args.temporal_window)
            _save_npz(save_path, tokens[0])
            processed += 1
        except Exception as exc:
            failed += 1
            logging.exception("Rank {} failed on {}: {}", rank, filepath, exc)

    print(
        "[rank {}] Finished depth encoding: processed={}, failed={}".format(
            rank, processed, failed
        ),
        flush=True,
    )


def main():
    args = _parse_args()
    rank, world_size, local_rank = _resolve_rank_info(args.local_rank)
    device, visible_gpu_count = _resolve_device(args.device, local_rank)

    filepaths = _list_depth_files(args.video_pattern, recursive=args.recursive)
    if not filepaths:
        raise FileNotFoundError("No npz files found under {}".format(args.video_pattern))

    pending_files, ready_files = _collect_pending_files(filepaths, args)
    shard_files = pending_files[rank::world_size]
    print(
        "[rank {}] world_size={}, local_rank={}, visible_gpus={}, device={}, total_files={}, ready_outputs={}, pending_files={}, assigned_files={}".format(
            rank,
            world_size,
            local_rank,
            visible_gpu_count,
            device,
            len(filepaths),
            ready_files,
            len(pending_files),
            len(shard_files),
        ),
        flush=True,
    )

    tokenizer = _build_tokenizer(args, device)
    _process_files(tokenizer, shard_files, args, rank)


if __name__ == "__main__":
    main()
