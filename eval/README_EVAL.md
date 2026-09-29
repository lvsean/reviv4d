# Egocentric hand reconstruction benchmark

Point the script at three folders and it reproduces the hand rows of the
paper: GA-MPJPE, RA-MPJPE and PA-MPJPE in millimetres, plus the cost per clip.

**We evaluate on every clip of each validation set, without any filtering.**
No clip or frame is dropped because a hand is occluded, leaves the field of
view, is only partly visible or is missed by a detector, and the evaluation is
not restricted to frames with a minimum number of visible joints. Some methods
report numbers on such filtered subsets. We consider the unfiltered protocol
the right one for the egocentric setting: with a head-mounted camera,
occlusion by the manipulated object and by the other hand, and hands moving
in and out of the frame, are unavoidable and make up a large part of the data.
They are the normal case rather than the exception, so a benchmark that
removes them measures a different, easier task than the one a method faces on
real egocentric video.

## 1. Checkpoints

Download one checkpoint set (see [Pretrained models](../README.md#pretrained-models))
and keep its files together in one folder:

```
reviv_checkpoints/reviv_500b/
  reviv_main.pth            # the any-to-any model
  reviv_tok_lhand.pth       # left-hand VQ-VAE (detokenizer)
  reviv_tok_rhand.pth       # right-hand VQ-VAE (detokenizer)
  norm_stats/
    lhand_mean.npy  lhand_std.npy
    rhand_mean.npy  rhand_std.npy
```

Pass the folder with `--ckpt_root` (or export `REVIV_CKPT_ROOT`). The
detokenizers and norm stats must come from the same set as `reviv_main.pth`.

Download the Cosmos video tokenizer as described in
[Inference](../README.md#inference); the 256-pathway sets (`reviv_500b`,
`metric_depth`) use `Cosmos/checkpoints/Cosmos-0.1-Tokenizer-DV4x8x8/encoder.jit`,
which is the default `--cosmos_dir`.

## 2. Data

Two folders: the clips and their ground truth.

**Clips** (`--video_dir`): 2 s egocentric RGB videos, any resolution and frame
rate (`.mp4 .mov .avi .mkv .webm .m4v`). Put every clip of one dataset in one
folder, or use one sub-folder per dataset and pass `--recursive`:

```
clips/
  taco/00000-0.mp4  00000-1.mp4 ...
  arctic/s05_box_grab_01_00-0.mp4 ...
  hot3d_aria/P0001_10a27bf7_0-0.mp4 ...
  holoassist/R0027-12-GoPro_0-0.mp4 ...
```

The first 2 s of each clip are resampled exactly as in `demo_hand.py`
(16 frames at 8 fps, 256 x 256 centre crop).

**Ground truth** (`--gt_dir`): one `.npz` per clip, mirroring the clip folders:

```
gt_joints/
  taco/00000.npz ...
  arctic/s05_box_grab_01_00.npz ...
```

Each file holds `lhand` and `rhand`, float arrays `[T, 21, 3]`: 21 joints
per hand, wrist first, in the camera frame of the clip, in metres, at 30 fps.
Either

* `T = 60` and the file is named like the clip (`<stem>.npz`), or
* `T = 120` and the clips are named `<base>-0` / `<base>-1`, which take frames
  `0-59` and `60-119` of `<base>.npz` (the layout of the EgoM2P label files).

## 3. Run

From the repository root:

```bash
python eval/eval_hand.py \
    --video_dir clips --recursive \
    --gt_dir    gt_joints \
    --ckpt_root reviv_checkpoints/reviv_500b \
    --output_dir eval_hand_output
```

Options: `--mode single` (one forward pass per clip instead of the paper's
chained MaskGIT schedule, ~4x faster), `--batch_size` (default 8 for a 24 GB
GPU), `--no_sdpa` (the repository's original attention; then use
`--batch_size 1`). Re-running with the same `--output_dir` resumes.

## 4. Protocol

The evaluation is the one used for the paper's table:

* Generation: `tok_rgb` + `rgb@256` conditioning, `tok_rhand` then `tok_lhand`
  in a chained MaskGIT schedule (2 steps, temperature 0.01, CFG 3.0, top_p 0.8,
  seed 0), decoded with the hand VQ-VAEs.
* Metrics per clip and hand over the 60 output frames: RA-MPJPE subtracts the
  wrist per frame; GA-MPJPE fits one similarity transform (rotation,
  translation, scale) to the whole clip; PA-MPJPE fits it per frame. The two
  hands are averaged, then all clips.
