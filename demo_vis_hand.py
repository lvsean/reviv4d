"""Overlay predicted / GT hand joints on the input clip.

For each clip folder produced by demo_hand.py this script

  1. loads the input RGB clip ``<stem>.mp4`` (16 frames, 8 fps, 256x256),
  2. loads the predicted hand joints ``<stem>_tok_{l,r}hand.npy`` and, when
     present, the GT hand joints ``<stem>_gt_tok_{l,r}hand.npy`` (both are
     camera-space [60, 21, 3] at 30 fps), and
  3. projects the joints into the clip with a pinhole camera and writes a
     30-fps overlay video ``<stem>_hand_vis.mp4``.

The RGB clip keeps its native 8 fps: every motion frame ``i`` (30 fps) reuses
RGB frame ``floor(i * rgb_fps / 30)``, so the background advances at 8 fps
while the hand skeletons animate at the full 30 fps.

GT hands are drawn in red, predictions in blue (both hands share the group
color, so GT vs prediction is the only distinction).

Camera model: the joints are in the optical frame (X right, Y down, Z forward,
metres), so accurate projection needs the intrinsics of the camera that shot
each clip. Intrinsics are resolved in this order: ``--geocalib`` (estimate the
intrinsics per clip from the RGB frame with GeoCalib) > an explicit/stored
intrinsic (``--gt_intrinsic`` or a ``<stem>_gt_intrinsic.npy`` sidecar) > a
known-dataset intrinsic (H2O clips, recognized from names like
``subject3_ego-k2-0_03``) > ``--focal`` > a fallback focal (1.1 * max(H, W))
that overlays the GT hands well on the bundled ARCTIC example clips.

Examples:
    # one clip folder
    python demo_vis_hand.py --input ./demo_output_hand/s05_box_use_01_02-1

    # every clip folder under an output root
    python demo_vis_hand.py --input ./demo_output_hand

    # GT and prediction in separate side-by-side panels
    python demo_vis_hand.py --input ./demo_output_hand --side_by_side

    # estimate the camera intrinsics from each clip with GeoCalib
    python demo_vis_hand.py --input ./demo_output_hand --geocalib
"""

import argparse
import os
import shutil
import subprocess

import cv2
import numpy as np

# Hand joint order (see tmp/overlay_hand_on_video.py): 0 wrist; 1-3 index;
# 4-6 middle; 7-9 pinky; 10-12 ring; 13-15 thumb; fingertips 16-20 =
# thumb, index, middle, ring, pinky. Each finger chain wrist -> ... -> tip.
HAND_FINGERS = [
    [0, 13, 14, 15, 16],   # thumb
    [0, 1, 2, 3, 17],      # index
    [0, 4, 5, 6, 18],      # middle
    [0, 10, 11, 12, 19],   # ring
    [0, 7, 8, 9, 20],      # pinky
]
HAND_BONES = [(f[i], f[i + 1]) for f in HAND_FINGERS for i in range(len(f) - 1)]

MOTION_FPS = 30.0  # hand joints are predicted at 30 fps (60 frames / 2 s)
DEFAULT_RGB_FPS = 8.0  # the 256x256 input clip is 16 frames / 2 s

# The joints are metric (metres) in camera space, so the correct focal length
# is the one of the camera that shot each dataset's clips. Only some datasets
# (e.g. nymeria) ship intrinsic sidecars; when none is available this scale
# (fx = fy = scale * max(H, W), ~50-degree FOV) is used, which overlays the GT
# hands well on the bundled ARCTIC example clips. Override with --focal.
DEFAULT_FOCAL_SCALE = 1.1

# BGR colors (cv2): GT in red, prediction in blue (both hands share the group
# color so GT vs prediction is the only distinction).
COLORS = {"gt": (0, 0, 255), "pred": (255, 0, 0)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project predicted / GT hand joints onto the input clip."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="A single clip folder (holding <stem>.mp4 and <stem>_tok_*.npy) "
        "or an output root whose subfolders are all such clip folders.",
    )
    parser.add_argument(
        "--focal",
        type=float,
        default=None,
        help="Pinhole focal length in pixels (fx = fy), measured on the input "
        "clip resolution. Default: 0.5 * max(H, W) (~90-degree FOV).",
    )
    parser.add_argument(
        "--gt_intrinsic",
        default=None,
        help="Optional path to a [3,3] or [T,3,3] intrinsic .npy; if given it "
        "overrides --focal and is scaled to the clip resolution.",
    )
    parser.add_argument(
        "--geocalib",
        action="store_true",
        help="Estimate each clip's intrinsics from its RGB frame with GeoCalib "
        "(https://github.com/cvg/GeoCalib) instead of using a stored/known/"
        "fallback intrinsic. Takes precedence over all other intrinsic options. "
        "Requires the `geocalib` package.",
    )
    parser.add_argument(
        "--geocalib_frame",
        type=int,
        default=0,
        help="Which RGB frame index GeoCalib estimates the intrinsics from.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=3,
        help="Integer upscale factor for the output video (256 -> 256*scale).",
    )
    parser.add_argument(
        "--side_by_side",
        action="store_true",
        help="Render GT and prediction in two panels instead of overlaying "
        "them on one frame.",
    )
    parser.add_argument(
        "--separation",
        type=float,
        default=6.0,
        help="Display-only horizontal offset (in clip pixels) pushing GT and "
        "prediction apart when overlaid, so near-identical hands don't fully "
        "cover each other. 0 disables. Ignored with --side_by_side.",
    )
    parser.add_argument(
        "--point_radius",
        type=float,
        default=1.6,
        help="Joint dot radius as a multiple of --scale.",
    )
    parser.add_argument(
        "--line_thickness",
        type=float,
        default=0.55,
        help="Bone line thickness as a multiple of --scale.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=MOTION_FPS,
        help="Output video frame rate (defaults to the 30 fps motion rate).",
    )
    return parser.parse_args()


def load_rgb_clip(path):
    """Return (frames [N,H,W,3] uint8 RGB, fps)."""
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_RGB_FPS
    frames = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"No RGB frames decoded from {path}")
    return np.stack(frames), (fps if fps > 0 else DEFAULT_RGB_FPS)


_GEOCALIB_MODEL = None


def _get_geocalib_model():
    """Load and cache the GeoCalib model (lazily, only when --geocalib is set)."""
    global _GEOCALIB_MODEL
    if _GEOCALIB_MODEL is None:
        try:
            import torch
            from geocalib import GeoCalib
        except ImportError as exc:
            raise ImportError(
                "--geocalib needs the `geocalib` package "
                "(https://github.com/cvg/GeoCalib). Install it, or drop "
                "--geocalib to use a stored/known/fallback intrinsic."
            ) from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _GEOCALIB_MODEL = GeoCalib().to(device)
    return _GEOCALIB_MODEL


def geocalib_intrinsics(rgb_frame):
    """Estimate (fx, fy, cx, cy) from one RGB frame with GeoCalib.

    rgb_frame is [H, W, 3] uint8 RGB; the returned intrinsics are already at
    that resolution (GeoCalib calibrates at the input image size).
    """
    import torch

    model = _get_geocalib_model()
    device = next(model.parameters()).device
    img = torch.from_numpy(np.ascontiguousarray(rgb_frame))
    img = img.permute(2, 0, 1).float().div(255.0).to(device)
    with torch.no_grad():
        result = model.calibrate(img)
    cam = result["camera"]
    fx, fy = cam.f[0].tolist()
    cx, cy = cam.c[0].tolist()
    return float(fx), float(fy), float(cx), float(cy)


def h2o_intrinsics(height, width):
    """Effective intrinsics for the H2O clips.

    The provided calibration is for the original 1280x720 frame; the clips are
    the centre 720x720 crop resized to the model resolution. Cropping shifts
    the principal point (cx -= (1280-720)/2, cy unchanged) and the resize scales
    fx, fy, cx, cy by <clip size> / 720.
    """
    fx, fy, cx, cy = 636.65930176, 636.25195312, 635.28387451, 366.87402344
    crop = 720
    cx -= (1280 - crop) / 2.0
    sx, sy = width / crop, height / crop
    return fx * sx, fy * sy, cx * sx, cy * sy


def is_h2o_clip(stem):
    """H2O clip names look like ``subject3_ego-k2-0_03`` (see the reference)."""
    return stem.startswith("subject") and "ego" in stem


def load_intrinsic_file(path, height, width):
    """Load a [3,3] or [T,3,3] intrinsic .npy and scale it to the clip size."""
    matrices = np.load(path)
    matrix = matrices[0] if matrices.ndim == 3 else matrices
    # The principal point encodes the source resolution (cx ~ (W_src - 1) / 2).
    src_w = 2.0 * float(matrix[0, 2]) + 1.0
    src_h = 2.0 * float(matrix[1, 2]) + 1.0
    sx, sy = width / src_w, height / src_h
    return (
        float(matrix[0, 0]) * sx,
        float(matrix[1, 1]) * sy,
        float(matrix[0, 2]) * sx,
        float(matrix[1, 2]) * sy,
    )


def build_intrinsics(height, width, focal, gt_intrinsic_path, stem=""):
    """Resolve (fx, fy, cx, cy) for a clip.

    Precedence: explicit/stored intrinsic (``--gt_intrinsic`` or a sidecar) >
    known-dataset intrinsic (H2O by name) > ``--focal`` > the ARCTIC fallback.
    """
    if gt_intrinsic_path is not None:
        return load_intrinsic_file(gt_intrinsic_path, height, width)
    if is_h2o_clip(stem):
        return h2o_intrinsics(height, width)
    if focal is not None:
        return focal, focal, 0.5 * (width - 1), 0.5 * (height - 1)
    fx = fy = DEFAULT_FOCAL_SCALE * max(height, width)
    print(f"  [warn] no intrinsics found; assuming fx=fy={fx:.0f}px "
          f"({DEFAULT_FOCAL_SCALE}x max(H,W), tuned for the ARCTIC examples). "
          "Pass --focal or --gt_intrinsic for other cameras.")
    return fx, fy, 0.5 * (width - 1), 0.5 * (height - 1)


def project(joints, intr):
    """Project camera-space joints [...,3] to pixel coords [...,2]."""
    fx, fy, cx, cy = intr
    z = np.clip(joints[..., 2], 1e-3, None)
    u = fx * joints[..., 0] / z + cx
    v = fy * joints[..., 1] / z + cy
    return np.stack([u, v], axis=-1)


def draw_hand(frame, pts2d, color, line_th, radius):
    """Draw one hand's bones + joints (pts2d: [21, 2] in output pixels)."""
    for a, b in HAND_BONES:
        pa = tuple(np.round(pts2d[a]).astype(int))
        pb = tuple(np.round(pts2d[b]).astype(int))
        cv2.line(frame, pa, pb, color, line_th, cv2.LINE_AA)
    for p in pts2d:
        cv2.circle(frame, tuple(np.round(p).astype(int)), radius, color, -1, cv2.LINE_AA)


def load_hands(clip_dir, stem, prefix):
    """Load {'lhand': [60,21,3], 'rhand': ...} for 'pred' ('') or 'gt' groups."""
    tag = "_gt" if prefix == "gt" else ""
    hands = {}
    for hand in ("lhand", "rhand"):
        path = os.path.join(clip_dir, f"{stem}{tag}_tok_{hand}.npy")
        if os.path.isfile(path):
            hands[hand] = np.load(path)
    return hands


def draw_legend(frame, groups, scale):
    """Small color key in the top-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.35 * scale
    y = int(14 * scale * 0.6)
    for group in groups:
        cv2.putText(frame, group.upper(), (int(4 * scale), y), font, fs,
                    COLORS[group], max(1, scale // 2), cv2.LINE_AA)
        y += int(12 * scale * 0.6)


def reencode_h264(src, dst):
    """Re-encode src to H.264 / yuv420p at dst so browsers and the VS Code
    video preview (both Chromium-based, H.264-only) can play it.

    OpenCV's mp4v writer produces MPEG-4 Part 2, which those players reject.
    Falls back to keeping the mp4v file if ffmpeg is unavailable.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        if os.path.abspath(src) != os.path.abspath(dst):
            os.replace(src, dst)
        print("  [warn] ffmpeg not found; leaving mp4v-encoded video (may not "
              "preview in VS Code / browsers).")
        return
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", src,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst],
        check=True,
    )
    if os.path.abspath(src) != os.path.abspath(dst):
        os.remove(src)


def render_clip(clip_dir, stem, args):
    video_path = os.path.join(clip_dir, f"{stem}.mp4")
    if not os.path.isfile(video_path):
        print(f"  skip {stem}: no input clip {video_path}")
        return
    pred = load_hands(clip_dir, stem, "pred")
    gt = load_hands(clip_dir, stem, "gt")
    if not pred and not gt:
        print(f"  skip {stem}: no hand joint .npy files")
        return

    rgb, rgb_fps = load_rgb_clip(video_path)
    n_rgb, h, w = rgb.shape[:3]
    if args.geocalib:
        # Estimate this clip's intrinsics from an RGB frame with GeoCalib.
        frame_idx = min(max(args.geocalib_frame, 0), n_rgb - 1)
        intr = geocalib_intrinsics(rgb[frame_idx])
        print(f"  geocalib intrinsics (frame {frame_idx}): "
              f"fx={intr[0]:.1f} fy={intr[1]:.1f} cx={intr[2]:.1f} cy={intr[3]:.1f}")
    else:
        # Prefer an explicit --gt_intrinsic, then a per-clip sidecar (as copied
        # by demo_infer.py for datasets that have one); build_intrinsics then
        # falls back to the H2O-by-name intrinsic, --focal, or the ARCTIC default.
        intrinsic_path = args.gt_intrinsic
        if intrinsic_path is None:
            sidecar = os.path.join(clip_dir, f"{stem}_gt_intrinsic.npy")
            intrinsic_path = sidecar if os.path.isfile(sidecar) else None
        intr = build_intrinsics(h, w, args.focal, intrinsic_path, stem=stem)

    # Pre-project every hand once: {(group, hand): [T, 21, 2]}.
    n_motion = max(
        (v.shape[0] for group in (pred, gt) for v in group.values()),
        default=0,
    )
    # Display-only nudge: overlaid GT/pred are often near-identical, so push
    # them apart horizontally (GT left, pred right) by --separation clip pixels.
    half_sep = 0.0 if args.side_by_side else args.separation / 2.0
    group_dx = {"gt": -half_sep, "pred": +half_sep}
    projected = {}
    for group_name, group in (("pred", pred), ("gt", gt)):
        for hand, joints in group.items():
            pts = project(joints, intr)
            pts[..., 0] += group_dx[group_name]
            projected[(group_name, hand)] = pts * args.scale

    scale = args.scale
    line_th = max(1, round(args.line_thickness * scale))
    radius = max(1, round(args.point_radius * scale))
    out_h, out_w = h * scale, w * scale
    panels = 2 if args.side_by_side else 1
    out_path = os.path.join(clip_dir, f"{stem}_hand_vis.mp4")
    # OpenCV writes mp4v; re-encode to H.264 afterwards. Use a temp file so a
    # failed run never leaves an unplayable file at the final path.
    tmp_path = os.path.join(clip_dir, f"{stem}_hand_vis.mp4v.mp4")
    writer = cv2.VideoWriter(
        tmp_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (out_w * panels, out_h),
    )

    for i in range(n_motion):
        rgb_idx = min(n_rgb - 1, int(i * rgb_fps / MOTION_FPS))
        base = cv2.cvtColor(rgb[rgb_idx], cv2.COLOR_RGB2BGR)
        base = cv2.resize(base, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        if args.side_by_side:
            gt_panel, pred_panel = base.copy(), base.copy()
            panel_of = {"gt": gt_panel, "pred": pred_panel}
            for (group_name, hand), pts in projected.items():
                draw_hand(panel_of[group_name], pts[i], COLORS[group_name], line_th, radius)
            draw_legend(gt_panel, ["gt"], scale)
            draw_legend(pred_panel, ["pred"], scale)
            frame = np.concatenate([gt_panel, pred_panel], axis=1)
        else:
            frame = base
            for (group_name, hand), pts in projected.items():
                draw_hand(frame, pts[i], COLORS[group_name], line_th, radius)
            draw_legend(frame, [g for g in ("gt", "pred") if any(k[0] == g for k in projected)], scale)

        writer.write(frame)
    writer.release()
    reencode_h264(tmp_path, out_path)
    groups = "+".join(g for g in ("gt", "pred") if any(k[0] == g for k in projected))
    print(f"  wrote {stem}_hand_vis.mp4 ({n_motion} frames, {groups})")


def find_clip_dirs(root):
    """A folder with an <stem>.mp4 is a clip dir; otherwise scan its subdirs."""
    stem = os.path.basename(os.path.normpath(root))
    if os.path.isfile(os.path.join(root, f"{stem}.mp4")):
        return [(root, stem)]
    clip_dirs = []
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, f"{name}.mp4")):
            clip_dirs.append((sub, name))
    return clip_dirs


def main():
    args = parse_args()
    clip_dirs = find_clip_dirs(args.input)
    if not clip_dirs:
        raise FileNotFoundError(
            f"No clip folders (with <stem>.mp4) found under {args.input}. Run "
            "demo_hand.py first so the input clip is copied next to the joints."
        )
    print(f"Rendering {len(clip_dirs)} clip(s)")
    for clip_dir, stem in clip_dirs:
        print(f"[{stem}]")
        render_clip(clip_dir, stem, args)


if __name__ == "__main__":
    main()
