"""ReViV demo inference: RGB video -> depth / camera / gaze / body.

Given a single RGB video, this script
  1. loads the pretrained NVIDIA Cosmos video tokenizer and encodes a 2 s clip
     into discrete tokens,
  2. feeds the tokens as condition to the trained ReViV any-to-any model, and
  3. predicts depth video, camera trajectory (`tok_cam`), eye gaze (`tok_gaze`)
     and full-body motion (`tok_body`) tokens, which are then detokenized and
     written to <output_dir>/<video_stem>/.

Two model flavors are supported; the pathway is picked automatically from the
domains stored in the main checkpoint (`args.all_domains`):
  * 512 ("metric depth") checkpoints condition on `tok_rgb_512` (a 32-frame
    512x512 clip at 16 fps, Cosmos DV8x16x16 -> 5x32x32 = 5120 tokens) and
    predict 512x512 depth (`tok_depth_512`).
  * 256 checkpoints (e.g. the stripped `checkpoint-119`) condition on
    `tok_rgb` (a 16-frame 256x256 clip at 8 fps, Cosmos DV4x8x8 -> 5120
    tokens) plus the raw normalized clip `rgb@256`, and can only predict
    256x256 depth (`tok_depth`, 16 frames).

--video accepts a single clip or a directory, in which case every *.mp4 inside
is processed (models are loaded once); each clip's predictions are written to
<output_dir>/<video_stem>/.

Besides the predictions, each clip's output folder also receives
  * a copy of the input RGB clip (<video_stem>.mp4), which demo_vis.py uses
    for point-cloud colors without needing a separate --video flag, and
  * GT camera sidecars <video_stem>_gt_cam.npy / <video_stem>_gt_intrinsic.npy
    when they are found next to the clip (same folder, a gt/ subfolder, or the
    example_data layout ../gt/<video_dir_name>/); demo_vis.py then offers GT
    intrinsic / GT first-pose visualization modes.

Example (checkpoint paths default to $REVIV_CKPT_ROOT/reviv_*.pth):
    python demo_infer.py --video /path/to/egocentric_clip.mp4 --output_dir ./demo_output
    python demo_infer.py --video /folder/of/clips/ --output_dir ./demo_output

    # point at a self-contained checkpoint set: reviv_*.pth are picked up from
    # the folder and the mean/std sidecars from its norm_stats/ subfolder
    python demo_infer.py \
        --video /path/to/egocentric_clip.mp4 \
        --ckpt_root /path/to/reviv_checkpoints/reviv \
        --output_dir ./demo_output

    # or with explicit checkpoints:
    python demo_infer.py \
        --video /path/to/egocentric_clip.mp4 \
        --checkpoint /path/to/reviv_main.pth \
        --body_tokenizer /path/to/reviv_tok_body.pth \
        --cam_tokenizer  /path/to/reviv_tok_cam.pth \
        --gaze_tokenizer /path/to/reviv_tok_gaze.pth \
        --output_dir ./demo_output

Requirements:
  * Cosmos checkpoints (encoder.jit / decoder.jit), downloadable with
        python cosmos_tokenizer/download_cosmos_tokenizer.py \
            --repo_id nvidia/Cosmos-1.0-Tokenizer-DV8x16x16 \
            --output_dir Cosmos/checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16
    (512 pathway; the 256 pathway uses Cosmos-0.1-Tokenizer-DV4x8x8 instead).
  * The cam / body detokenizers denormalize with mean/std .npy sidecars. They
    are looked up under <ckpt_root>/norm_stats/ if that folder exists,
    otherwise under the folder given by the $REVIV_DATA_ROOT env var, see
    README_DATA.md.
"""

import argparse
import glob
import os
import shutil
from contextlib import nullcontext

import numpy as np
import torch
from tokenizers import Tokenizer
from decord import VideoReader, cpu

from run_training_reviv import get_model as get_reviv_model
from run_training_vqvae import get_model as get_tok_model
from reviv.models.generate import (
    GenerationSampler,
    build_chained_generation_schedules,
    init_empty_target_modality,
    init_full_input_modality,
)
from reviv.data.modality_info import (
    MODALITY_INFO,
    apply_vocab_overrides,
    infer_token_vocab_sizes,
)
import reviv.utils.plotting_utils as plotting_utils
from reviv.utils.plotting_utils import decode_dict
from cosmos_tokenizer.video_lib import CausalVideoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_grad_enabled(False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Per-model-flavor settings. The pathway is chosen from the main checkpoint's
# `args.all_domains`: checkpoints trained with the 512 pathway carry
# tok_rgb_512 / tok_depth_512, the 256 ones tok_rgb / tok_depth.
PATHWAYS = {
    "512": {
        # The 512 models condition on the Cosmos tokens only.
        "cond_domains": ["tok_rgb_512"],
        # Raw-clip domain this pathway *could* take alongside the tokens (the
        # naming is asymmetric: rgb512 here, rgb@256 for the 256 pathway). It is
        # only fed when it appears in cond_domains above, so it is unused here.
        "raw_clip_domain": "rgb512",
        "target_tokens": {
            "tok_depth_512": 5120,  # 5 x 32 x 32 Cosmos depth tokens
            "tok_cam": 30,          # camera trajectory tokens
            "tok_gaze": 30,         # eye gaze tokens
            "tok_body": 210,        # full-body motion tokens (30 x 7)
        },
        # Input clip expected by the rgb512 pathway: 2 s at 16 fps, 512 x 512.
        "num_frames": 32,
        "clip_fps": 16,
        "frame_size": 512,
        "cosmos_dir": "Cosmos/checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16",
        "depth_domain": "tok_depth_512",
    },
    "256": {
        # The 256 models condition on the Cosmos tokens plus the raw clip.
        "cond_domains": ["tok_rgb", "rgb@256"],
        "raw_clip_domain": "rgb@256",
        "target_tokens": {
            "tok_depth": 5120,      # 5 x 32 x 32 Cosmos depth tokens (256x256)
            "tok_cam": 30,
            "tok_gaze": 30,
            "tok_body": 210,
        },
        # Input clip expected by the rgb256 pathway: 2 s at 8 fps, 256 x 256.
        "num_frames": 16,
        "clip_fps": 8,
        "frame_size": 256,
        "cosmos_dir": "Cosmos/checkpoints/Cosmos-0.1-Tokenizer-DV4x8x8",
        "depth_domain": "tok_depth",
    },
}

# tok_depth_512 <-> tok_depth, so --targets works regardless of the pathway.
DEPTH_DOMAINS = {pw["depth_domain"] for pw in PATHWAYS.values()}
ALL_TARGET_CHOICES = sorted({t for pw in PATHWAYS.values() for t in pw["target_tokens"]})


# Default checkpoint locations, overridable with the REVIV_CKPT_ROOT env var
CKPT_ROOT = os.environ.get("REVIV_CKPT_ROOT", "./reviv_checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="ReViV RGB-video -> depth/cam/gaze/body demo.")
    parser.add_argument(
        "--video",
        required=True,
        help="Input RGB video (any resolution / fps), or a directory whose "
        "*.mp4 files are all processed.",
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
    parser.add_argument("--body_tokenizer", default=None,
                        help="Body VQ-VAE tokenizer checkpoint (.pth). Default: <ckpt_root>/reviv_tok_body.pth.")
    parser.add_argument("--cam_tokenizer", default=None,
                        help="Camera-trajectory VQ-VAE tokenizer checkpoint (.pth). Default: <ckpt_root>/reviv_tok_cam.pth.")
    parser.add_argument("--gaze_tokenizer", default=None,
                        help="Gaze VQ-VAE tokenizer checkpoint (.pth). Default: <ckpt_root>/reviv_tok_gaze.pth.")
    parser.add_argument(
        "--cosmos_dir",
        default=None,
        help="Directory holding the Cosmos encoder.jit / decoder.jit. Defaults "
        "to Cosmos/checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16 for 512 "
        "checkpoints and Cosmos/checkpoints/Cosmos-0.1-Tokenizer-DV4x8x8 "
        "for 256 checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        default="./demo_output",
        help="Root output directory; predictions go to <output_dir>/<video_stem>/.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        choices=ALL_TARGET_CHOICES,
        help="Modalities to predict (default: all the loaded checkpoint "
        "supports). tok_depth / tok_depth_512 are mapped to whichever depth "
        "resolution the checkpoint provides.",
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


def load_clip(video_path, start_sec=0.0, num_frames=32, clip_fps=16, frame_size=512):
    """Read a 2 s clip as uint8 [1, T, H, W, 3] in [0..255], square-cropped."""
    import cv2

    vr = VideoReader(video_path, ctx=cpu(0))
    native_fps = vr.get_avg_fps()
    if not native_fps or not np.isfinite(native_fps):
        print(f"Warning: {video_path} reports no frame rate; assuming {clip_fps} fps.")
        native_fps = clip_fps
    start = int(round(start_sec * native_fps))
    # Sample num_frames frames at clip_fps from the native frame rate.
    indices = start + np.round(np.arange(num_frames) * native_fps / clip_fps).astype(int)
    if indices[0] >= len(vr):
        raise ValueError(f"--start_sec {start_sec} is beyond the end of {video_path}")
    if indices[-1] > len(vr) - 1:
        print(
            f"Warning: clip needs frames up to index {indices[-1]} but {video_path} "
            f"has only {len(vr)}; the last frame will be repeated."
        )
    indices = np.clip(indices, 0, len(vr) - 1)
    frames = vr.get_batch(indices).asnumpy()  # [T, H, W, 3] uint8

    # Resize the short side to frame_size, then center-crop to frame_size^2.
    t, h, w, _ = frames.shape
    scale = frame_size / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = np.stack([cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames])
    top, left = (new_h - frame_size) // 2, (new_w - frame_size) // 2
    clip = resized[:, top:top + frame_size, left:left + frame_size]
    return clip[None]  # [1, T, H, W, 3]


def encode_rgb_tokens(video_paths, cosmos_dir, start_sec=0.0,
                      num_frames=32, clip_fps=16, frame_size=512, keep_clips=False):
    """Tokenize RGB clips with one Cosmos DV encoder.

    Returns {path: {"tokens": [1, 5120] int64 (cpu),
                    "clip": [1, T, H, W, 3] float in [-1, 1] (cpu) or None}}.
    The normalized clip is kept only when keep_clips is set (the 256 pathway
    conditions on the raw clip as `rgb@256` in addition to the tokens).
    """
    encoder_ckpt = os.path.join(cosmos_dir, "encoder.jit")
    if not os.path.isfile(encoder_ckpt):
        raise FileNotFoundError(
            f"{encoder_ckpt} not found. Download it with "
            "cosmos_tokenizer/download_cosmos_tokenizer.py (see module docstring)."
        )
    encoder = CausalVideoTokenizer(checkpoint_enc=encoder_ckpt, device=DEVICE)
    encoded_by_video = {}
    for video_path in video_paths:
        print(f"Loading clip from {video_path} ...")
        clip = load_clip(video_path, start_sec=start_sec, num_frames=num_frames,
                         clip_fps=clip_fps, frame_size=frame_size)
        print(f"Encoding {clip.shape} clip with the Cosmos tokenizer ...")
        with torch.inference_mode():
            tokens = encoder(clip, temporal_window=num_frames)  # [1, 5, 32, 32]
        tokens = torch.as_tensor(np.asarray(tokens), dtype=torch.int64)
        entry = {"tokens": tokens.reshape(1, -1).cpu(), "clip": None}  # [1, 5120]
        if keep_clips:
            entry["clip"] = torch.from_numpy(clip).float().div(255).mul(2).sub(1)
        encoded_by_video[video_path] = entry
    del encoder
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return encoded_by_video


def apply_ckpt_root(args, ckpt_files):
    """Resolve checkpoint paths and norm stats from a single --ckpt_root folder.

    ckpt_files maps argparse attribute names to filenames under ckpt_root
    (e.g. {"checkpoint": "reviv_main.pth"}); attributes already set via their
    explicit flags are left untouched. If <ckpt_root>/norm_stats/ exists, the
    mean/std .npy sidecars are read from there; an explicitly set
    REVIV_DATA_ROOT env var still takes precedence.
    """
    for attr, filename in ckpt_files.items():
        if getattr(args, attr) is None:
            setattr(args, attr, os.path.join(args.ckpt_root, filename))
    norm_dir = os.path.join(args.ckpt_root, "norm_stats")
    if "REVIV_DATA_ROOT" not in os.environ and os.path.isdir(norm_dir):
        plotting_utils.DATA_ROOT = norm_dir
        print(f"Using normalization stats from {norm_dir}")


def _check_ckpt_exists(ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Point --ckpt_root (or the "
            "REVIV_CKPT_ROOT env var) at the directory holding the reviv_*.pth "
            "checkpoints, or pass --checkpoint / --body_tokenizer / "
            "--cam_tokenizer / --gaze_tokenizer explicitly."
        )


def load_motion_tokenizer(ckpt_path):
    _check_ckpt_exists(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = get_tok_model(ckpt["args"], DEVICE)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def load_main_model(ckpt_path):
    """Load the main any-to-any model; returns (model, all_domains)."""
    _check_ckpt_exists(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = ckpt["args"]
    # Checkpoints saved before the fourm -> reviv rename store the old
    # registry builder name (fm_*); map it to the renamed builder.
    if model_args.model.startswith("fm_"):
        model_args.model = "reviv_" + model_args.model[len("fm_"):]
    if not hasattr(model_args, "all_domains"):
        raise RuntimeError(
            "Checkpoint args are missing 'all_domains'. Use a checkpoint saved by "
            "run_training_reviv.py (its args carry the in/out domain lists)."
        )
    modality_info = {mod: MODALITY_INFO[mod] for mod in model_args.all_domains}
    # Build the token embeddings / output heads at the checkpoint's codebook
    # sizes so variants with a different vocab load cleanly (e.g. reviv_500b
    # trains tok_body at 1024 instead of the default 2048).
    vocab_sizes = infer_token_vocab_sizes(ckpt["model"])
    overridden = {m: vocab_sizes[m] for m in modality_info
                  if m in vocab_sizes and vocab_sizes[m] != MODALITY_INFO[m].get("vocab_size")}
    if overridden:
        print(f"Overriding vocab sizes from checkpoint: {overridden}")
    modality_info = apply_vocab_overrides(modality_info, vocab_sizes)
    model = get_reviv_model(model_args, modality_info)
    model.load_state_dict(ckpt["model"])
    return model.eval().to(DEVICE), list(model_args.all_domains)


def select_pathway(all_domains):
    """Pick the 512 or 256 pathway from the checkpoint's domain list."""
    pathway = "512" if "tok_rgb_512" in all_domains else "256"
    missing = [d for d in PATHWAYS[pathway]["cond_domains"] if d not in all_domains]
    if missing:
        raise RuntimeError(
            f"Checkpoint domains {all_domains} miss the {pathway} pathway "
            f"conditions {missing}; this demo cannot drive the model."
        )
    return pathway


def resolve_targets(requested, pathway, all_domains):
    """Map requested targets onto the checkpoint's domains (depth res swap)."""
    pw = PATHWAYS[pathway]
    if requested is None:
        return [t for t in pw["target_tokens"] if t in all_domains]
    targets = []
    for target in requested:
        if target in DEPTH_DOMAINS and target not in pw["target_tokens"]:
            mapped = pw["depth_domain"]
            print(f"Note: {target} is not available in this checkpoint; "
                  f"predicting {mapped} instead.")
            target = mapped
        if target not in all_domains:
            raise ValueError(f"Target {target} is not supported by this checkpoint "
                             f"(available domains: {all_domains}).")
        if target not in targets:
            targets.append(target)
    return targets


def build_sample(cond_tensors, target_domain, num_tokens, text_tok):
    """Condition modalities (tokens and/or raw video) + one empty target."""
    sample = {}
    for cond_domain, tensor in cond_tensors.items():
        cond_ntoks = MODALITY_INFO[cond_domain]["max_tokens"]
        sample[cond_domain] = {
            "tensor": tensor.to(DEVICE),
            "input_mask": torch.zeros(1, cond_ntoks, dtype=torch.bool, device=DEVICE),
            "target_mask": torch.ones(1, cond_ntoks, dtype=torch.bool, device=DEVICE),
        }
    sample = init_empty_target_modality(sample, MODALITY_INFO, target_domain, 1, num_tokens, DEVICE)
    for cond_domain in cond_tensors:
        sample = init_full_input_modality(
            sample, MODALITY_INFO, cond_domain, DEVICE, eos_id=text_tok.token_to_id("[EOS]")
        )
    return sample


def build_schedule(cond_domains, target_domain, num_tokens):
    return build_chained_generation_schedules(
        cond_domains=cond_domains,
        target_domains=[target_domain],
        tokens_per_target=[num_tokens],
        autoregression_schemes=["roar"],
        decoding_steps=[3],
        token_decoding_schedules=["linear"],
        temps=[0.01],
        temp_schedules=["constant"],
        cfg_scales=[2.0],
        cfg_schedules=["constant"],
        cfg_grow_conditioning=True,
    )


def generation_context(amp_dtype):
    if DEVICE != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


# GT sidecar kinds -> accepted filename suffixes (both spellings are found in
# the wild; copies are always written with the canonical first suffix).
GT_SIDECAR_SUFFIXES = {
    "gt_cam": ("_gt_cam.npy", "_first_frame_cam.npy"),
    "gt_intrinsic": ("_gt_intrinsic.npy", "_gt_intri.npy", "_intrinsic.npy"),
}


def find_gt_sidecars(video_path):
    """Find GT camera pose / intrinsic .npy files that belong to a clip.

    Both the full video stem and the stem without a trailing ``_rgb512`` are
    tried, in the video's own folder as well as ``<video_dir>/gt/`` and the
    example_data layout ``<video_dir>/../gt/<video_dir_name>/``.
    Returns {kind: path} for the sidecars that exist.
    """
    video_path = os.path.abspath(video_path)
    video_dir = os.path.dirname(video_path)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    stems = [video_stem]
    if video_stem.endswith("_rgb512"):
        stems.append(video_stem[: -len("_rgb512")])
    search_dirs = [
        video_dir,
        os.path.join(video_dir, "gt"),
        os.path.join(os.path.dirname(video_dir), "gt", os.path.basename(video_dir)),
    ]
    found = {}
    for kind, suffixes in GT_SIDECAR_SUFFIXES.items():
        candidates = (
            os.path.join(directory, stem + suffix)
            for directory in search_dirs
            for stem in stems
            for suffix in suffixes
        )
        found[kind] = next((c for c in candidates if os.path.isfile(c)), None)
    return {kind: path for kind, path in found.items() if path is not None}


def copy_clip_and_gt_sidecars(video_path, output_dir, video_stem):
    """Mirror the RGB clip and any GT cam/intrinsic files into output_dir.

    demo_vis.py picks up <stem>.mp4 for point-cloud colors and the
    <stem>_gt_{cam,intrinsic}.npy sidecars for its GT visualization modes.
    """
    video_copy = os.path.join(output_dir, f"{video_stem}.mp4")
    if os.path.abspath(video_path) != os.path.abspath(video_copy):
        shutil.copy2(video_path, video_copy)
        print(f"Copied RGB clip to {video_copy}")
    for kind, src in find_gt_sidecars(video_path).items():
        dst = os.path.join(output_dir, f"{video_stem}_{kind}.npy")
        shutil.copy2(src, dst)
        print(f"Copied GT sidecar: {src} -> {dst}")


def main():
    args = parse_args()
    apply_ckpt_root(args, {
        "checkpoint": "reviv_main.pth",
        "body_tokenizer": "reviv_tok_body.pth",
        "cam_tokenizer": "reviv_tok_cam.pth",
        "gaze_tokenizer": "reviv_tok_gaze.pth",
    })
    if os.path.isdir(args.video):
        video_paths = sorted(glob.glob(os.path.join(args.video, "*.mp4")))
        if not video_paths:
            raise FileNotFoundError(f"No *.mp4 files found in {args.video}")
        print(f"Found {len(video_paths)} videos in {args.video}")
    else:
        video_paths = [args.video]

    # 1. Load the ReViV main model and pick the matching input pathway
    text_tok = Tokenizer.from_file(args.text_tokenizer)
    print(f"Loading ReViV model from {args.checkpoint} ...")
    model, all_domains = load_main_model(args.checkpoint)
    sampler = GenerationSampler(model)
    pathway = select_pathway(all_domains)
    pw = PATHWAYS[pathway]
    print(f"Checkpoint uses the {pathway} pathway "
          f"(conditions: {pw['cond_domains']}, depth: {pw['depth_domain']}).")
    targets = resolve_targets(args.targets, pathway, all_domains)
    print(f"Predicting: {targets}")

    # 2. RGB videos -> Cosmos tokens (one shared encoder instance)
    cosmos_dir = args.cosmos_dir or pw["cosmos_dir"]
    encoded_by_video = encode_rgb_tokens(
        video_paths, cosmos_dir, args.start_sec,
        num_frames=pw["num_frames"], clip_fps=pw["clip_fps"],
        frame_size=pw["frame_size"], keep_clips=pw["raw_clip_domain"] in pw["cond_domains"],
    )

    # 3. Load the detokenizers once
    toks = {}
    if "tok_body" in targets:
        toks["tok_body"] = load_motion_tokenizer(args.body_tokenizer)
    if "tok_cam" in targets:
        toks["tok_cam"] = load_motion_tokenizer(args.cam_tokenizer)
    if "tok_gaze" in targets:
        toks["tok_gaze"] = load_motion_tokenizer(args.gaze_tokenizer)
    if pw["depth_domain"] in targets:
        toks[pw["depth_domain"]] = CausalVideoTokenizer(
            checkpoint_dec=os.path.join(cosmos_dir, "decoder.jit"), device=DEVICE
        )

    # 4. Per video: predict each target modality separately (memory friendly),
    #    then decode into <output_dir>/<video_stem>/.
    for video_index, video_path in enumerate(video_paths):
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.output_dir, video_stem)
        os.makedirs(output_dir, exist_ok=True)
        # Keep the RGB clip and any GT cam/intrinsic sidecars next to the
        # predictions so demo_vis.py finds everything in one folder.
        copy_clip_and_gt_sidecars(video_path, output_dir, video_stem)
        filename = video_stem + ".npz"
        encoded = encoded_by_video[video_path]
        cond_tensors = {pw["cond_domains"][0]: encoded["tokens"]}
        if pw["raw_clip_domain"] in pw["cond_domains"]:
            cond_tensors[pw["raw_clip_domain"]] = encoded["clip"]
        print(f"[{video_index + 1}/{len(video_paths)}] {video_stem}")

        for target_domain in targets:
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            num_tokens = pw["target_tokens"][target_domain]
            print(f"Generating {pw['cond_domains']} -> {target_domain} ({num_tokens} tokens) ...")

            sample = build_sample(cond_tensors, target_domain, num_tokens, text_tok)
            schedule = build_schedule(pw["cond_domains"], target_domain, num_tokens)
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

            # Detokenize the prediction and write it under output_dir
            # (the conditions are skipped inside decode_dict).
            decode_dict(
                filename,
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
