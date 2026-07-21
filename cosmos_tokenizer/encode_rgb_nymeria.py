"""Multi-node RGB tokenization for Nymeria videos.

Launch this with one Slurm task per node and one torchrun worker per GPU, e.g.

    python -m torch.distributed.run \
        --nproc_per_node=4 \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_PROCID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        -m cosmos_tokenizer.encode_rgb_nymeria ...

The script shards the full video list across the global world size and builds
one tokenizer instance per worker.
"""

import os
from argparse import ArgumentParser
from typing import NamedTuple

import mediapy as mp
import numpy as np
import torch
from loguru import logger as logging

from cosmos_tokenizer.networks import TokenizerConfigs
from cosmos_tokenizer.video_lib import CausalVideoTokenizer


class VideoJob(NamedTuple):
    output_subdir: str
    seq_name: str
    filepath: str
    filename: str


def _parse_args():
    parser = ArgumentParser(description="Multi-node Nymeria RGB tokenizer.")
    parser.add_argument(
        "--video_pattern",
        type=str,
        required=True,
        help="Input directory containing mp4 files, Nymeria sequences, or a single mp4 file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where token npz files will be written.",
    )
    parser.add_argument(
        "--video_subdir",
        type=str,
        default="videos_512_16fps",
        help="Per-sequence subdirectory that contains the mp4 files.",
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
        default=None,
        help="Encoder checkpoint filepath.",
    )
    parser.add_argument(
        "--checkpoint_dec",
        type=str,
        default=None,
        help="Decoder checkpoint filepath.",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["CV", "DV"],
        default=None,
        help="Tokenizer type. Required when --mode=torch.",
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
        default="jit",
        help="Tokenizer backend.",
    )
    parser.add_argument(
        "--temporal_window",
        type=int,
        default=32,
        help="Temporal window size passed to the tokenizer.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Tokenizer inference dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Requested device. Default uses the Slurm-assigned GPU.",
    )
    parser.add_argument(
        "--frames_per_chunk",
        type=int,
        default=32,
        help="Frames per tokenization chunk.",
    )
    parser.add_argument(
        "--chunks_per_video",
        type=int,
        default=2,
        help="How many chunks each video is split into before tokenization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for mp4 files under --video_pattern.",
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
        raise RuntimeError("CUDA is required for encode_rgb_nymeria.py.")

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
        if args.tokenizer_type is None:
            raise ValueError("--tokenizer_type is required when --mode=torch")
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


def _infer_seq_name(filepath, video_subdir):
    parent_dir = os.path.basename(os.path.dirname(filepath))
    if parent_dir == video_subdir:
        return os.path.basename(os.path.dirname(os.path.dirname(filepath)))
    return parent_dir


def _looks_like_month_dir(dirname):
    return len(dirname) == 6 and dirname.isdigit()


def _infer_output_subdir(root_path, filepath, video_subdir):
    root_base = os.path.basename(os.path.normpath(root_path))
    if _looks_like_month_dir(root_base):
        return root_base

    parent_dir = os.path.basename(os.path.dirname(filepath))
    if parent_dir != video_subdir:
        return ""

    seq_dir = os.path.dirname(os.path.dirname(filepath))
    month_dir = os.path.basename(os.path.dirname(seq_dir))
    if _looks_like_month_dir(month_dir):
        return month_dir

    if not os.path.isfile(root_path):
        rel_path = os.path.relpath(filepath, root_path)
        rel_parts = rel_path.split(os.sep)
        if rel_parts and _looks_like_month_dir(rel_parts[0]):
            return rel_parts[0]

    return ""


def _list_video_jobs(root_path, video_subdir, recursive):
    jobs = []
    if os.path.isfile(root_path):
        if (
            root_path.endswith(".mp4")
            and os.path.basename(os.path.dirname(root_path)) == video_subdir
        ):
            jobs.append(
                VideoJob(
                    output_subdir=_infer_output_subdir(root_path, root_path, video_subdir),
                    seq_name=_infer_seq_name(root_path, video_subdir),
                    filepath=root_path,
                    filename=os.path.basename(root_path),
                )
            )
        return jobs

    if recursive:
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames.sort()
            if os.path.basename(dirpath) != video_subdir:
                continue
            for filename in sorted(filenames):
                if filename.endswith(".mp4"):
                    filepath = os.path.join(dirpath, filename)
                    jobs.append(
                        VideoJob(
                            output_subdir=_infer_output_subdir(root_path, filepath, video_subdir),
                            seq_name=_infer_seq_name(filepath, video_subdir),
                            filepath=filepath,
                            filename=filename,
                        )
                    )
        jobs.sort(key=lambda job: job.filepath)
        return jobs

    direct_files = sorted(
        filename for filename in os.listdir(root_path) if filename.endswith(".mp4")
    )
    if direct_files and os.path.basename(os.path.normpath(root_path)) == video_subdir:
        for filename in direct_files:
            filepath = os.path.join(root_path, filename)
            jobs.append(
                VideoJob(
                    output_subdir=_infer_output_subdir(root_path, filepath, video_subdir),
                    seq_name=_infer_seq_name(filepath, video_subdir),
                    filepath=filepath,
                    filename=filename,
                )
            )
        return jobs

    for seq_name in sorted(os.listdir(root_path)):
        seq_path = os.path.join(root_path, seq_name)
        if not os.path.isdir(seq_path):
            continue

        video_dir = os.path.join(seq_path, video_subdir)
        if not os.path.isdir(video_dir):
            continue

        for filename in sorted(os.listdir(video_dir)):
            if filename.endswith(".mp4"):
                jobs.append(
                    VideoJob(
                        output_subdir=_infer_output_subdir(root_path, os.path.join(video_dir, filename), video_subdir),
                        seq_name=seq_name,
                        filepath=os.path.join(video_dir, filename),
                        filename=filename,
                    )
                )
    return jobs


def _build_output_paths(output_dir, output_subdir, seq_name, filename, chunks_per_video):
    target_dir = output_dir
    if output_subdir:
        target_dir = os.path.join(output_dir, output_subdir)
    stem = "{}_{}".format(os.path.splitext(filename)[0], seq_name)
    return [
        os.path.join(target_dir, "{}_{}.npz".format(stem, chunk_idx))
        for chunk_idx in range(chunks_per_video)
    ]


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


def _collect_pending_jobs(jobs, args):
    pending_jobs = []
    ready_jobs = 0

    for job in jobs:
        output_paths = _build_output_paths(
            args.output_dir,
            job.output_subdir,
            job.seq_name,
            job.filename,
            args.chunks_per_video,
        )
        if not args.overwrite and all(_has_valid_npz(path) for path in output_paths):
            ready_jobs += 1
            continue
        pending_jobs.append((job, output_paths))

    return pending_jobs, ready_jobs


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


def _load_video_chunks(filepath, frames_per_chunk, chunks_per_video):
    video = np.asarray(mp.read_video(filepath))
    expected_frames = frames_per_chunk * chunks_per_video
    if video.ndim != 4:
        raise ValueError("Expected [T,H,W,C] video array, got {}".format(video.shape))
    if video.shape[0] != expected_frames:
        raise ValueError(
            "Expected {} frames in {}, got {}".format(
                expected_frames, filepath, video.shape[0]
            )
        )
    return video.reshape(
        chunks_per_video,
        frames_per_chunk,
        video.shape[1],
        video.shape[2],
        video.shape[3],
    )


def _process_jobs(tokenizer, pending_jobs, rank, args):
    processed = 0
    failed = 0

    os.makedirs(args.output_dir, exist_ok=True)
    for job, output_paths in pending_jobs:
        try:
            chunks = _load_video_chunks(
                job.filepath,
                frames_per_chunk=args.frames_per_chunk,
                chunks_per_video=args.chunks_per_video,
            )
            if args.overwrite:
                pending_chunk_indices = list(range(len(output_paths)))
            else:
                pending_chunk_indices = [
                    chunk_idx
                    for chunk_idx, output_path in enumerate(output_paths)
                    if not _has_valid_npz(output_path)
                ]
            if not pending_chunk_indices:
                continue

            with torch.inference_mode():
                for chunk_idx in pending_chunk_indices:
                    output_path = output_paths[chunk_idx]
                    tokens = tokenizer(
                        chunks[chunk_idx : chunk_idx + 1],
                        temporal_window=args.temporal_window,
                    )
                    _save_npz(output_path, tokens[0])
            processed += 1
        except Exception as exc:
            failed += 1
            logging.exception("Rank {} failed on {}: {}", rank, job.filepath, exc)

    print(
        "[rank {}] Finished RGB encoding: processed={}, failed={}".format(rank, processed, failed),
        flush=True,
    )


def main():
    args = _parse_args()
    rank, world_size, local_rank = _resolve_rank_info(args.local_rank)
    device, visible_gpu_count = _resolve_device(args.device, local_rank)

    jobs = _list_video_jobs(args.video_pattern, args.video_subdir, args.recursive)
    if not jobs:
        raise FileNotFoundError("No mp4 files found under {}".format(args.video_pattern))

    pending_jobs, ready_jobs = _collect_pending_jobs(jobs, args)
    shard_jobs = pending_jobs[rank::world_size]
    print(
        "[rank {}] world_size={}, local_rank={}, visible_gpus={}, device={}, total_videos={}, ready_outputs={}, pending_videos={}, assigned_videos={}".format(
            rank,
            world_size,
            local_rank,
            visible_gpu_count,
            device,
            len(jobs),
            ready_jobs,
            len(pending_jobs),
            len(shard_jobs),
        ),
        flush=True,
    )

    tokenizer = _build_tokenizer(args, device)
    _process_jobs(tokenizer, shard_jobs, rank, args)


if __name__ == "__main__":
    main()
