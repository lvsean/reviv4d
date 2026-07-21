"""ReViV demo inference: RGB video -> left / right hand motion.

Given a single RGB video, this script
  1. loads the pretrained NVIDIA Cosmos video tokenizer and encodes a 2 s clip
     into discrete RGB tokens. Hands are conditioned on the 256 pathway
     (`tok_rgb` + the raw `rgb@256` clip, 16-frame 256x256, DV4x8x8), which all
     released checkpoints (reviv / reviv_500b / metric_depth) carry; the 512
     `tok_rgb_512` pathway is only used for a checkpoint that lacks the 256
     conditioning domains,
  2. feeds the condition(s) to the trained ReViV any-to-any model, and
  3. predicts right- and left-hand motion tokens (`tok_rhand` / `tok_lhand`,
     30x7 = 210 each), which are detokenized into camera-space hand joints
     [60, 21, 3] at 30 fps and written to <output_dir>/<video_stem>/.

Besides the predictions, each clip's output folder also receives
  * a copy of the input RGB clip (<video_stem>.mp4), used by demo_vis_hand.py
    as the projection background, and
  * GT hand joints <video_stem>_gt_tok_{l,r}hand.npy [60, 21, 3] copied
    straight from the GT joints file (canonical layout
    <video_dir>/../gt_joints/<dataset>/<stem>.npz, keys `lhand` / `rhand`);
    demo_vis_hand.py then overlays GT and prediction. GT is never obtained by
    detokenizing GT tokens.

Works with any checkpoint that predicts tok_lhand / tok_rhand and carries a
supported RGB pathway (tok_rgb + rgb@256, or tok_rgb_512). The reviv,
reviv_500b and metric_depth checkpoints all qualify; the token embeddings are
rebuilt at each checkpoint's codebook sizes (e.g. reviv_500b's tok_body 1024)
so they load cleanly.

--video accepts a single clip or a directory, in which case every *.mp4 inside
is processed (models are loaded once).

Example (checkpoint paths default to $REVIV_CKPT_ROOT/reviv_*.pth):
    python demo_hand.py --video /path/to/egocentric_clip.mp4 --output_dir ./demo_output

    # point at a self-contained checkpoint set: reviv_*.pth are picked up from
    # the folder and the mean/std sidecars from its norm_stats/ subfolder
    python demo_hand.py \
        --video /path/to/egocentric_clip.mp4 \
        --ckpt_root /path/to/reviv_checkpoints/reviv \
        --output_dir ./demo_output

    # or with explicit checkpoints:
    python demo_hand.py \
        --video /path/to/egocentric_clip.mp4 \
        --checkpoint /path/to/reviv_main.pth \
        --lhand_tokenizer /path/to/reviv_tok_lhand.pth \
        --rhand_tokenizer /path/to/reviv_tok_rhand.pth \
        --output_dir ./demo_output

Requirements:
  * The Cosmos DV4x8x8 encoder (encoder.jit), downloadable with
        python cosmos_tokenizer/download_cosmos_tokenizer.py \
            --repo_id nvidia/Cosmos-0.1-Tokenizer-DV4x8x8 \
            --output_dir Cosmos/checkpoints/Cosmos-0.1-Tokenizer-DV4x8x8
  * The hand detokenizers denormalize with the {lhand,rhand}_{mean,std}.npy
    sidecars, looked up under <ckpt_root>/norm_stats/ if that folder exists,
    otherwise under the folder given by the $REVIV_DATA_ROOT env var.
"""

import argparse
import os
import shutil

import numpy as np
import torch
from tokenizers import Tokenizer

from demo_infer import (
    CKPT_ROOT,
    DEVICE,
    PATHWAYS,
    apply_ckpt_root,
    encode_rgb_tokens,
    generation_context,
    load_main_model,
    load_motion_tokenizer,
)
from reviv.models.generate import (
    GenerationSampler,
    build_chained_generation_schedules,
    init_empty_target_modality,
    init_full_input_modality,
)
from reviv.data.modality_info import MODALITY_INFO
from reviv.utils.plotting_utils import decode_dict

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_grad_enabled(False)

TARGET_DOMAINS = ["tok_rhand", "tok_lhand"]
TOKENS_PER_TARGET = [210, 210]  # 30 x 7 motion tokens per hand

# The RGB conditioning (which tok_rgb domain, clip length/fps/size and Cosmos
# encoder) is taken from demo_infer.PATHWAYS and chosen per checkpoint: the 256
# checkpoints (reviv / reviv_500b) condition on tok_rgb + rgb@256, the 512
# "metric depth" checkpoint on tok_rgb_512. Only the hand targets are fixed.

# Video extensions collected when --video points at a directory.
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

# GT hand *joints* (already decoded, [60, 21, 3] per hand) live in a single
# per-clip .npz with `lhand` / `rhand` keys, under the example_data layout
# <video_dir>/../gt_joints/<dataset>/<stem>.npz. These are copied verbatim; GT
# is never obtained by detokenizing GT tokens.


def parse_args():
    parser = argparse.ArgumentParser(description="ReViV RGB-video -> hand-motion demo.")
    parser.add_argument(
        "--video",
        required=True,
        help="Input RGB video (any resolution / fps), or a directory whose "
        "video files are all processed. Supported extensions: "
        + ", ".join(VIDEO_EXTS) + ".",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When --video is a directory, also search subdirectories for "
        "video files.",
    )
    parser.add_argument(
        "--ckpt_root",
        default=CKPT_ROOT,
        help="Folder holding the reviv_*.pth checkpoints and, optionally, a "
        "norm_stats/ subfolder with the mean/std .npy sidecars. Defaults to "
        "$REVIV_CKPT_ROOT (fallback ./reviv_checkpoints).",
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Main ReViV model checkpoint (.pth). Default: <ckpt_root>/reviv_main.pth.")
    parser.add_argument("--lhand_tokenizer", default=None,
                        help="Left-hand VQ-VAE tokenizer checkpoint (.pth). Default: <ckpt_root>/reviv_tok_lhand.pth.")
    parser.add_argument("--rhand_tokenizer", default=None,
                        help="Right-hand VQ-VAE tokenizer checkpoint (.pth). Default: <ckpt_root>/reviv_tok_rhand.pth.")
    parser.add_argument(
        "--cosmos_dir",
        default=None,
        help="Directory holding the Cosmos encoder.jit. Defaults to the "
        "pathway's encoder: Cosmos-0.1-Tokenizer-DV4x8x8 for 256 checkpoints, "
        "Cosmos-1.0-Tokenizer-DV8x16x16 for the 512 (metric depth) checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        default="./demo_output",
        help="Root output directory; predictions go to <output_dir>/<video_stem>/.",
    )
    parser.add_argument("--start_sec", type=float, default=0.0, help="Clip start time in the input video.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=float, default=0.0)
    parser.add_argument(
        "--amp_dtype",
        default="bf16",
        choices=["none", "bf16", "fp16"],
        help="Autocast dtype for generation. bf16/fp16 reduce attention memory.",
    )
    parser.add_argument(
        "--text_tokenizer",
        default="./reviv/utils/tokenizer/trained/text_tokenizer_reviv_wordpiece_30k.json",
    )
    return parser.parse_args()


def collect_videos(path, recursive=False):
    """Resolve --video into a sorted list of video file paths.

    A single file is returned as-is; a directory is scanned for files whose
    extension is in VIDEO_EXTS (case-insensitive), optionally recursing.
    """
    if not os.path.isdir(path):
        return [path]

    if recursive:
        paths = [
            os.path.join(root, name)
            for root, _, files in os.walk(path)
            for name in files
            if name.lower().endswith(VIDEO_EXTS)
        ]
    else:
        paths = [
            entry.path
            for entry in os.scandir(path)
            if entry.is_file() and entry.name.lower().endswith(VIDEO_EXTS)
        ]
    return sorted(paths)


def find_gt_joints_file(video_path):
    """Locate the GT hand-joints .npz that belongs to a clip, or None.

    GT joints are already decoded ([60, 21, 3] per hand) and live in a single
    per-clip .npz with `lhand` / `rhand` keys. The example_data layout
    ``<video_dir>/../gt_joints/<dataset>/<stem>.npz`` is tried first, then a
    couple of flatter fallbacks.
    """
    video_path = os.path.abspath(video_path)
    video_dir = os.path.dirname(video_path)
    dataset = os.path.basename(video_dir)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    candidates = [
        os.path.join(os.path.dirname(video_dir), "gt_joints", dataset, stem + ".npz"),
        os.path.join(video_dir, "gt_joints", stem + ".npz"),
        os.path.join(os.path.dirname(video_dir), "gt_joints", stem + ".npz"),
    ]
    return next((c for c in candidates if os.path.isfile(c)), None)


def copy_inputs_and_gt(video_path, output_dir, video_stem):
    """Mirror the input clip and the GT hand joints into output_dir.

    Writes <stem>.mp4 (the input clip, for demo_vis_hand.py's background) and,
    when the GT joints .npz is found, <stem>_gt_tok_{l,r}hand.npy holding the GT
    joints [60, 21, 3] read straight from the file — the same format as the
    predictions so demo_vis_hand.py can overlay both. GT is never obtained by
    detokenizing GT tokens.
    """
    video_copy = os.path.join(output_dir, f"{video_stem}.mp4")
    if os.path.abspath(video_path) != os.path.abspath(video_copy):
        shutil.copy2(video_path, video_copy)
        print(f"Copied input clip to {video_copy}")

    src = find_gt_joints_file(video_path)
    if src is None:
        return
    npz = np.load(src)
    for hand in ("lhand", "rhand"):
        if hand not in npz.files:
            continue
        dst = os.path.join(output_dir, f"{video_stem}_gt_tok_{hand}.npy")
        np.save(dst, np.asarray(npz[hand]))  # [60, 21, 3]
    print(f"Copied GT joints from {src}")


def select_hand_pathway(all_domains):
    """Pick the RGB conditioning pathway for hand prediction.

    Unlike demo_infer.select_pathway (which prefers the 512 depth pathway),
    hands are conditioned on the 256 pathway (tok_rgb + rgb@256) across all
    released checkpoints — including metric_depth, whose 512 pathway is meant
    for depth, not hands. Conditioning metric_depth's hands on tok_rgb_512
    gives much worse joints (the 256 example clips would also have to be
    upscaled to 512 and padded to 32 frames). So prefer 256, and fall back to
    512 only for a checkpoint that lacks the 256 conditioning domains.
    """
    for name in ("256", "512"):
        if all(d in all_domains for d in PATHWAYS[name]["cond_domains"]):
            return name
    raise RuntimeError(
        f"Checkpoint domains {all_domains} carry neither the 256 "
        f"({PATHWAYS['256']['cond_domains']}) nor the 512 "
        f"({PATHWAYS['512']['cond_domains']}) RGB conditioning pathway."
    )


def build_sample(encoded, text_tok, pw):
    """RGB condition(s) for the pathway + both empty hand targets."""
    cond_tensors = {}
    for cond_domain in pw["cond_domains"]:
        # tok_rgb / tok_rgb_512 -> Cosmos tokens; rgb@256 -> the raw clip.
        cond_tensors[cond_domain] = (
            encoded["clip"] if cond_domain.startswith("rgb@") else encoded["tokens"]
        )
    sample = {}
    for cond_domain, tensor in cond_tensors.items():
        cond_ntoks = MODALITY_INFO[cond_domain]["max_tokens"]
        sample[cond_domain] = {
            "tensor": tensor.to(DEVICE),
            "input_mask": torch.zeros(1, cond_ntoks, dtype=torch.bool, device=DEVICE),
            "target_mask": torch.ones(1, cond_ntoks, dtype=torch.bool, device=DEVICE),
        }
    for target_domain, num_tokens in zip(TARGET_DOMAINS, TOKENS_PER_TARGET):
        sample = init_empty_target_modality(sample, MODALITY_INFO, target_domain, 1, num_tokens, DEVICE)
    for cond_domain in cond_tensors:
        sample = init_full_input_modality(
            sample, MODALITY_INFO, cond_domain, DEVICE, eos_id=text_tok.token_to_id("[EOS]")
        )
    return sample


def build_schedule(pw):
    # Same settings as the rgb -> lrhand evaluation: MaskGIT with 2 cosine
    # decoding steps per hand and a cfg scale of 3.
    return build_chained_generation_schedules(
        cond_domains=pw["cond_domains"],
        target_domains=TARGET_DOMAINS,
        tokens_per_target=TOKENS_PER_TARGET,
        autoregression_schemes=["maskgit"] * len(TARGET_DOMAINS),
        decoding_steps=[2] * len(TARGET_DOMAINS),
        token_decoding_schedules=["cosine"] * len(TARGET_DOMAINS),
        temps=[0.01] * len(TARGET_DOMAINS),
        temp_schedules=["constant"] * len(TARGET_DOMAINS),
        cfg_scales=[3.0] * len(TARGET_DOMAINS),
        cfg_schedules=["constant"] * len(TARGET_DOMAINS),
        cfg_grow_conditioning=True,
    )


def main():
    args = parse_args()
    apply_ckpt_root(args, {
        "checkpoint": "reviv_main.pth",
        "lhand_tokenizer": "reviv_tok_lhand.pth",
        "rhand_tokenizer": "reviv_tok_rhand.pth",
    })
    video_paths = collect_videos(args.video, recursive=args.recursive)
    if os.path.isdir(args.video):
        if not video_paths:
            raise FileNotFoundError(
                f"No video files ({', '.join(VIDEO_EXTS)}) found in {args.video}"
            )
        print(f"Found {len(video_paths)} videos in {args.video}")

    # 1. Load the ReViV main model, pick the matching input pathway, and make
    #    sure the checkpoint can predict hands.
    text_tok = Tokenizer.from_file(args.text_tokenizer)
    print(f"Loading ReViV model from {args.checkpoint} ...")
    model, all_domains = load_main_model(args.checkpoint)
    pathway = select_hand_pathway(all_domains)
    pw = PATHWAYS[pathway]
    missing = [d for d in pw["cond_domains"] + TARGET_DOMAINS if d not in all_domains]
    if missing:
        raise RuntimeError(
            f"Checkpoint domains {all_domains} miss {missing}; this demo needs a "
            f"checkpoint with the {pathway} pathway conditions {pw['cond_domains']} "
            "and tok_lhand / tok_rhand targets."
        )
    print(f"Checkpoint uses the {pathway} pathway (conditions: {pw['cond_domains']}).")
    sampler = GenerationSampler(model)

    # 2. RGB videos -> Cosmos tokens (+ normalized clips for the 256 pathway)
    cosmos_dir = args.cosmos_dir or pw["cosmos_dir"]
    encoded_by_video = encode_rgb_tokens(
        video_paths, cosmos_dir, args.start_sec,
        num_frames=pw["num_frames"], clip_fps=pw["clip_fps"], frame_size=pw["frame_size"],
        keep_clips="rgb@256" in pw["cond_domains"],
    )

    # 3. Hand detokenizers
    toks = {
        "tok_lhand": load_motion_tokenizer(args.lhand_tokenizer),
        "tok_rhand": load_motion_tokenizer(args.rhand_tokenizer),
    }

    # 4. Per video: predict both hands in one chained pass, then decode into
    #    <output_dir>/<video_stem>/<video_stem>_tok_{l,r}hand.npy ([60, 21, 3]).
    #    The input clip and (when available) GT hand joints are mirrored next
    #    to the predictions so demo_vis_hand.py finds everything in one folder.
    schedule = build_schedule(pw)
    for video_index, video_path in enumerate(video_paths):
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.output_dir, video_stem)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[{video_index + 1}/{len(video_paths)}] {video_stem}")

        copy_inputs_and_gt(video_path, output_dir, video_stem)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        sample = build_sample(encoded_by_video[video_path], text_tok, pw)
        with generation_context(args.amp_dtype):
            out_dict = sampler.generate(
                sample,
                schedule,
                text_tokenizer=text_tok,
                verbose=False,
                seed=args.seed,
                top_p=args.top_p,
                top_k=args.top_k,
            )

        # Detokenize + denormalize both hands (conditions are skipped inside
        # decode_dict).
        decode_dict(
            video_stem + ".npz",
            out_dict,
            toks,
            text_tok,
            image_size=256,
            patch_size=8,
            decoding_steps=50,
            output_dir=output_dir,
        )
        del sample, out_dict
        print(f"Done. Predictions written to {output_dir}")


if __name__ == "__main__":
    main()
