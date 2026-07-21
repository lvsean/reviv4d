#!/usr/bin/env python3
"""Interactive 3D visualization of demo_infer.py predictions.

Loads the ``*_tok_cam.npy``, ``*_tok_body.npy``, ``*_tok_gaze.npy`` and the
depth array written by demo_infer.py plus the original RGB clip (colors only)
and shows them in a viser scene: an RGB-colored depth point cloud, the
predicted camera frustum, the body skeleton, and the gaze ray to its
depth-selected 3D point.

Two depth conventions are auto-detected from the files present. The 512x512
metric-depth clips write ``*_tok_depth_512.npy`` (16 fps) and use the linear
0-15 m decode. The 256x256 relative-depth clips write ``*_tok_depth.npy``
(8 fps) and use the inverse-depth decode plus the relative-depth floor fit (see
tmp/vis_nymeria_new_0515.py). Body/cam/gaze always play at 30 fps; the RGB/depth
frames are held at their native clip rate (8 or 16 fps).

Camera calibration defaults to --camera-calibration auto, which picks the best
available option in this order:

1. 'GT first pose + GT intrinsic' -- used when demo_infer.py found GT sidecars
   (*_gt_intrinsic.npy / *_gt_cam.npy) next to the clip and copied them into the
   output folder. It anchors the trajectory to the GT first camera pose and
   reprojects with the GT intrinsics.
2. 'GeoCalib first-frame' -- estimates the first-frame gravity direction and
   focal length from the RGB frame and corrects the trajectory roll/pitch
   (relative motion is preserved).
3. The raw normalized 90-degree pinhole prediction, if GeoCalib is unavailable.

Pass --camera-calibration none / geocalib-first / gt-first-pose (or use the
sidebar) to force a specific mode. Floor-fitted affine depth is enabled by
default; unchecking it restores the raw decode.

--prediction_dir may point at the demo_infer.py output root holding many
<video_stem>/ subfolders; every sequence found is listed in the sidebar's
Dataset dropdown and can be loaded without restarting by hand. The RGB clip
copied by demo_infer.py into each output folder is picked up automatically,
so --video is only needed for predictions made before that change.

Example:
    python demo_vis.py --prediction_dir ./demo_output
    python demo_vis.py --prediction_dir ./demo_output --filename 000000_20230831_s1_ronald_harris_act4_zqf4xn_1_rgb512 --camera-calibration gt-first-pose

GeoCalib must be importable (see environment.yaml: it is installed from a
local source checkout, e.g. ``git clone https://github.com/cvg/GeoCalib.git``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import viser
import viser.transforms as tf
from scipy.spatial.transform import Rotation


MOTION_FPS = 30.0  # body / cam / gaze are predicted at 30 fps (60 frames / 2 s)
# The RGB/depth clip runs at 16 fps for the 512x512 metric-depth inputs and at
# 8 fps for the 256x256 relative-depth inputs. Body/cam/gaze always play at 30 fps.
CLIP_FPS_METRIC = 16.0    # 512x512 metric depth (32 frames / 2 s)
CLIP_FPS_RELATIVE = 8.0   # 256x256 relative depth (16 frames / 2 s)
NORMALIZED_FOV_DEG = 90.0
METRIC_DEPTH_MAX_M = 15.0
# Inverse-depth decode range for the 256x256 relative-depth inputs (see the
# floor_fitting reference in tmp/vis_nymeria_new_0515.py).
RELATIVE_DEPTH_DECODE_MIN_MM = 200.0
RELATIVE_DEPTH_DECODE_MAX_MM = 3800.0
MODE_PREDICTION = "Prediction (default)"
MODE_GEOCALIB = "GeoCalib first-frame"
MODE_GT_POSE_INTRINSIC = "GT first pose + GT intrinsic"
# The GT-first-pose mode is appended at runtime when the *_gt_{cam,intrinsic}.npy
# sidecars (copied by demo_infer.py) are present next to the predictions.
BASE_VISUALIZATION_MODES = (MODE_PREDICTION, MODE_GEOCALIB)

# Fixed convention change:
# camera optical (right, down, forward) -> body canonical (right, forward, up).
BODY_FROM_CAMERA_AXES = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

T2M_BONES = np.asarray(
    [
        [0, 2], [2, 5], [5, 8], [8, 11],
        [0, 1], [1, 4], [4, 7], [7, 10],
        [0, 3], [3, 6], [6, 9], [9, 12], [12, 15],
        [9, 14], [14, 17], [17, 19], [19, 21],
        [9, 13], [13, 16], [16, 18], [18, 20],
    ],
    dtype=np.int32,
)


def cone_mesh_from_segments(
    segments: np.ndarray,
    radius: float,
    radial_segments: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a tapered bone mesh, one cone per skeleton segment."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    angles = np.linspace(0.0, 2.0 * np.pi, radial_segments, endpoint=False, dtype=np.float32)
    circle = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    for start, end in np.asarray(segments, dtype=np.float32):
        axis = end - start
        length = float(np.linalg.norm(axis))
        if length < 1e-6:
            continue
        direction = axis / length
        helper = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(direction @ helper)) > 0.95:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        tangent = np.cross(direction, helper)
        tangent /= max(float(np.linalg.norm(tangent)), 1e-8)
        bitangent = np.cross(direction, tangent)
        base = len(vertices)
        ring = start + radius * (
            circle[:, [0]] * tangent[None] + circle[:, [1]] * bitangent[None]
        )
        vertices.extend(ring.tolist())
        tip = len(vertices)
        vertices.append(end.tolist())
        center = len(vertices)
        vertices.append(start.tolist())
        for index in range(radial_segments):
            nxt = (index + 1) % radial_segments
            faces.append([base + index, base + nxt, tip])
            faces.append([center, base + nxt, base + index])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


class ConeSkeleton:
    def __init__(
        self,
        server: viser.ViserServer,
        name: str,
        joints: np.ndarray,
        radius: float = 0.02,
    ) -> None:
        self.radius = float(radius)
        vertices, faces = cone_mesh_from_segments(joints[T2M_BONES], self.radius)
        self.node = server.scene.add_mesh_simple(
            name,
            vertices=vertices,
            faces=faces,
            color=(20, 90, 255),
            opacity=0.78,
            side="double",
        )

    def update(self, joints: np.ndarray) -> None:
        vertices, faces = cone_mesh_from_segments(joints[T2M_BONES], self.radius)
        self.node.vertices = vertices
        self.node.faces = faces

    @property
    def visible(self) -> bool:
        return bool(self.node.visible)

    @visible.setter
    def visible(self, value: bool) -> None:
        self.node.visible = bool(value)


def parse_stem(name: str) -> str:
    stem = Path(name).stem
    for suffix in ("_tok_cam", "_tok_depth_512", "_tok_depth", "_tok_body", "_tok_gaze", "_tok"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def sequence_indices(src_frames: int, dst_frames: int) -> np.ndarray:
    if src_frames < 1 or dst_frames < 1:
        raise ValueError(f"Invalid frame counts: src={src_frames}, dst={dst_frames}")
    return np.rint(np.linspace(0, src_frames - 1, dst_frames)).astype(np.int64)


def clip_indices_for_motion(clip_frames: int, motion_frames: int, clip_fps: float) -> np.ndarray:
    """Sample-and-hold map from the 30 fps motion timeline onto the clip (8/16 fps)."""
    times = np.arange(motion_frames, dtype=np.float64) / MOTION_FPS
    return np.minimum((times * clip_fps).astype(np.int64), clip_frames - 1)


def motion_indices_for_clip(motion_frames: int, clip_frames: int, clip_fps: float) -> np.ndarray:
    """Nearest 30 fps motion frame for each clip frame (8/16 fps)."""
    times = np.arange(clip_frames, dtype=np.float64) / clip_fps
    return np.minimum(np.rint(times * MOTION_FPS).astype(np.int64), motion_frames - 1)


def body_21_to_22(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected body [T,J,3], got {joints.shape}")
    if joints.shape[1] == 22:
        return joints
    if joints.shape[1] != 21:
        raise ValueError(f"Expected 21 or 22 body joints, got {joints.shape[1]}")
    out = np.empty((joints.shape[0], 22, 3), dtype=np.float32)
    out[:, :3] = joints[:, :3]
    out[:, 3] = joints[:, 0]
    out[:, 4:] = joints[:, 3:]
    return out


def decode_camera_9d(cam_9d: np.ndarray) -> np.ndarray:
    """Decode canonicalized 9D camera predictions directly into canonical c2w."""
    cam_9d = np.asarray(cam_9d, dtype=np.float32)
    if cam_9d.ndim == 3 and cam_9d.shape[1:] == (4, 4):
        poses = cam_9d.copy()
    else:
        if cam_9d.ndim != 2 or cam_9d.shape[1] != 9:
            raise ValueError(f"Expected predicted camera [T,9] or [T,4,4], got {cam_9d.shape}")
        packed = cam_9d.reshape(-1, 3, 3).transpose(0, 2, 1)
        a1, a2 = packed[:, :, 0], packed[:, :, 1]
        b1 = a1 / np.maximum(np.linalg.norm(a1, axis=1, keepdims=True), 1e-8)
        a2_orth = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
        b2 = a2_orth / np.maximum(np.linalg.norm(a2_orth, axis=1, keepdims=True), 1e-8)
        b3 = np.cross(b1, b2, axis=1)
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], cam_9d.shape[0], axis=0)
        poses[:, :3, :3] = np.stack((b1, b2, b3), axis=-1)
        poses[:, :3, 3] = packed[:, :, 2]

    return poses.astype(np.float32)


def camera_to_body_coordinates(cam_c2w: np.ndarray) -> np.ndarray:
    """Apply only the fixed optical-to-body axis convention change."""
    cam_c2w = np.asarray(cam_c2w, dtype=np.float32)
    if cam_c2w.ndim != 3 or cam_c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected camera poses [T,4,4], got {cam_c2w.shape}")
    return (BODY_FROM_CAMERA_AXES[None] @ cam_c2w).astype(np.float32)


def recon_cam_to_world(cam_9d: np.ndarray, gt_first_c2w: np.ndarray) -> np.ndarray:
    """Lift canonical camera predictions into the GT world frame.

    Mirrors ``recon_cam`` in tmp/vis_nymeria_new_0515.py: the canonicalized poses
    (frame 0 at the origin) are left-multiplied by the GT first-frame camera pose,
    so the predicted trajectory lives in the same world frame as the GT.
    """
    canonical = decode_camera_9d(cam_9d).astype(np.float64)
    gt_first_c2w = np.asarray(gt_first_c2w, dtype=np.float64)
    return np.einsum("ij,kjl->kil", gt_first_c2w, canonical).astype(np.float32)


def recover_points_to_world(points_cam: np.ndarray, gt_cam_w2c: np.ndarray) -> np.ndarray:
    """Map camera-space points (body joints) into the GT world frame.

    Mirrors ``recover_points`` in tmp/vis_nymeria_new_0515.py: the yaw of the GT
    first-frame camera rotation is zeroed before the points are mapped through
    ``w2c[0]``, which is what puts the body in the same frame as the camera
    trajectory returned by :func:`recon_cam_to_world`.
    """
    points_cam = np.asarray(points_cam, dtype=np.float64)
    if points_cam.ndim != 3 or points_cam.shape[2] != 3:
        raise ValueError(f"Expected points [T,N,3], got {points_cam.shape}")
    frames, num_points, _ = points_cam.shape
    flat = np.transpose(points_cam, (2, 0, 1)).reshape(3, -1)

    w2c = np.asarray(gt_cam_w2c, dtype=np.float64)
    if w2c.ndim != 3 or w2c.shape[1:] != (4, 4):
        raise ValueError(f"Expected GT camera poses [T,4,4], got {w2c.shape}")
    c2w_first = np.linalg.inv(w2c)[0]
    # Drop the yaw of the first camera so only gravity (pitch/roll) is applied.
    _, pitch, roll = Rotation.from_matrix(c2w_first[:3, :3]).as_euler("ZYX")
    gravity_rot = Rotation.from_euler("ZYX", [0.0, pitch, roll]).as_matrix()
    flat = np.linalg.inv(gravity_rot) @ flat

    flat = np.linalg.inv(w2c[0, :3, :3]) @ (flat - w2c[0, :3, 3].reshape(3, 1))
    world = np.transpose(flat.reshape(3, frames, num_points), (1, 2, 0))
    return world.astype(np.float32)


def load_raw_depth(path: Path) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    if depth.ndim == 4:
        # demo_infer writes [T,H,W,3] uint8 grayscale; collapse the channels.
        depth = depth.mean(axis=-1)
    if depth.ndim != 3:
        raise ValueError(f"Expected depth [T,H,W] or [T,H,W,3], got {depth.shape} at {path}")
    return depth.astype(np.float32)


def decode_original_metric_depth(raw_depth: np.ndarray) -> np.ndarray:
    """Decode the original metric-depth convention without floor fitting."""
    return (
        np.clip(np.asarray(raw_depth, dtype=np.float32), 0.0, 255.0)
        / 255.0
        * METRIC_DEPTH_MAX_M
    ).astype(np.float32)


def decode_relative_depth_to_meters(
    raw_depth: np.ndarray,
    min_depth_mm: float = RELATIVE_DEPTH_DECODE_MIN_MM,
    max_depth_mm: float = RELATIVE_DEPTH_DECODE_MAX_MM,
) -> np.ndarray:
    """Decode the 256x256 relative (inverse) depth into meters.

    Mirrors ``RelativeDepthLoader.decode_depth_to_meters`` from the
    tmp/vis_nymeria_new_0515.py reference: the raw 0-255 values encode inverse
    depth linearly between ``1/max_depth`` and ``1/min_depth``.
    """
    min_depth_mm = max(float(min_depth_mm), 1e-3)
    max_depth_mm = max(float(max_depth_mm), min_depth_mm + 1e-3)
    inv_max_depth = 1.0 / min_depth_mm
    inv_min_depth = 1.0 / max_depth_mm
    inv_depth_range = inv_max_depth - inv_min_depth
    raw = np.asarray(raw_depth, dtype=np.float32)
    depth = raw * inv_depth_range / 255.0 + inv_min_depth
    depth = 1.0 / np.maximum(depth, 1e-12)
    depth /= 1000.0
    return depth.astype(np.float32)


def load_rgb_video(path: Path, height: int, width: int) -> np.ndarray:
    """Load the RGB clip as uint8 colors (not used for any geometry estimate)."""
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[:2] != (height, width):
            frame_rgb = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(frame_rgb)
    capture.release()
    if not frames:
        raise ValueError(f"No RGB frames decoded from {path}")
    return np.stack(frames).astype(np.uint8, copy=False)


def normalized_rays(
    height: int,
    width: int,
    focal_xy: tuple[float, float] | None = None,
    principal_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return pinhole rays, using the normalized 90-degree convention by default."""
    if focal_xy is None:
        focal_xy = (0.5 * float(max(height, width)),) * 2
    fx, fy = map(float, focal_xy)
    if not np.isfinite(fx + fy) or fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid focal length: {(fx, fy)}")
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )
    if principal_xy is None:
        principal_xy = (0.5 * (width - 1), 0.5 * (height - 1))
    cx, cy = map(float, principal_xy)
    return np.stack(((u - cx) / fx, (v - cy) / fy, np.ones_like(u)), axis=-1)


def load_scaled_gt_intrinsics(
    path: Path, height: int, width: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Load GT intrinsics and scale the original image calibration to depth size."""
    matrices = np.load(path).astype(np.float32)
    matrix = matrices[0] if matrices.ndim == 3 else matrices
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected GT intrinsics [T,3,3] or [3,3], got {matrices.shape}")
    # Nymeria's principal point is at the image center (703.5 for 1408 px).
    source_width = 2.0 * float(matrix[0, 2]) + 1.0
    source_height = 2.0 * float(matrix[1, 2]) + 1.0
    scale_x, scale_y = width / source_width, height / source_height
    focal_xy = (float(matrix[0, 0] * scale_x), float(matrix[1, 1] * scale_y))
    principal_xy = (float(matrix[0, 2] * scale_x), float(matrix[1, 2] * scale_y))
    return focal_xy, principal_xy


def rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the shortest 3D rotation mapping ``source`` onto ``target``."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= max(float(np.linalg.norm(source)), 1e-12)
    target /= max(float(np.linalg.norm(target)), 1e-12)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if sine < 1e-8:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float32)
        axis = np.zeros(3, dtype=np.float64)
        axis[int(np.argmin(np.abs(source)))] = 1.0
        axis = np.cross(source, axis)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return (2.0 * np.outer(axis, axis) - np.eye(3)).astype(np.float32)
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]],
        dtype=np.float64,
    )
    return (np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)).astype(np.float32)


def apply_first_frame_gravity_calibration(
    cam_c2w: np.ndarray,
    gravity_camera: np.ndarray,
    world_up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Correct trajectory roll/pitch from first-frame camera-frame gravity.

    The minimal world rotation maps GeoCalib's frame-0 vertical direction to
    the body's predicted world-up. Applying it about the first camera center
    preserves frame-0 translation and all relative camera motion.
    """
    cam_c2w = np.asarray(cam_c2w, dtype=np.float32)
    gravity_camera = np.asarray(gravity_camera, dtype=np.float32).reshape(3)
    world_up = np.asarray(world_up, dtype=np.float32).reshape(3)
    predicted_vertical = cam_c2w[0, :3, :3] @ gravity_camera
    correction = rotation_between_vectors(predicted_vertical, world_up)
    calibrated = cam_c2w.copy()
    calibrated[:, :3, :3] = correction[None] @ calibrated[:, :3, :3]
    pivot = calibrated[0, :3, 3].copy()
    calibrated[:, :3, 3] = (cam_c2w[:, :3, 3] - pivot) @ correction.T + pivot
    angle_deg = float(np.rad2deg(np.arccos(np.clip((np.trace(correction) - 1.0) / 2.0, -1.0, 1.0))))
    return calibrated.astype(np.float32), correction, angle_deg


def run_geocalib_first_frame(
    rgb: np.ndarray,
    device_name: str,
    weights: str,
) -> tuple[np.ndarray, tuple[float, float], dict[str, float]]:
    """Estimate first-frame camera gravity and focal length with GeoCalib."""
    try:
        import torch
        from geocalib import GeoCalib
    except ImportError as exc:
        raise RuntimeError(
            "GeoCalib is unavailable; install it from a source checkout "
            "(git clone https://github.com/cvg/GeoCalib.git && pip install -e GeoCalib), "
            "see environment.yaml"
        ) from exc

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--geocalib-device cuda requested, but CUDA is unavailable")
    device = torch.device(device_name)
    model = GeoCalib(weights=weights).eval().to(device)
    image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0).to(device)
    result = model.calibrate(image, camera_model="pinhole")
    gravity = result["gravity"].vec3d[0].detach().cpu().numpy().astype(np.float32)
    focal = result["camera"].f[0].detach().cpu().numpy()
    uncertainties = {
        key: float(result[key].reshape(-1)[0].detach().cpu())
        for key in ("roll_uncertainty", "pitch_uncertainty", "focal_uncertainty")
        if key in result
    }
    return gravity, (float(focal[0]), float(focal[1])), uncertainties


def estimate_up_and_floor(body: np.ndarray, foot_offset_m: float = 0.10) -> tuple[np.ndarray, float]:
    """Estimate gravity-up from the predicted torso and floor from predicted feet."""
    torso_idx = 12 if body.shape[1] > 12 else body.shape[1] - 1
    torso = body[:, torso_idx] - body[:, 0]
    torso = torso[np.all(np.isfinite(torso), axis=1)]
    if torso.shape[0] == 0:
        raise ValueError("Cannot estimate up direction from predicted body")
    up = np.median(torso, axis=0)
    up /= max(float(np.linalg.norm(up)), 1e-8)

    foot_ids = [idx for idx in (10, 11) if idx < body.shape[1]]
    feet = body[:, foot_ids].reshape(-1, 3)
    feet = feet[np.all(np.isfinite(feet), axis=1)]
    if feet.shape[0] == 0:
        raise ValueError("Cannot estimate floor from predicted feet")
    floor_height = float(np.percentile(feet @ up, 10.0) - foot_offset_m)
    return up.astype(np.float32), floor_height


def fit_depth_to_predicted_floor(
    raw_depth: np.ndarray,
    provisional_metric_depth: np.ndarray,
    cam_c2w: np.ndarray,
    body: np.ndarray,
    focal_xy: tuple[float, float] | None,
    principal_xy: tuple[float, float] | None,
    floor_fraction: float = 0.1,
    debug: bool = False,
) -> tuple[float, float]:
    """Fit ``depth = a * raw + b`` so the lowest point-map values sit on the floor.

    The lowest ``floor_fraction`` of provisional point-map world-Z values are
    aligned to the median predicted foot height minus 0.2 m. ``provisional_metric_depth``
    is only used to rank floor candidates: for the 512x512 metric inputs it is the
    linear 0-15 m decode, for the 256x256 relative inputs it is the inverse-depth
    decode (see decode_relative_depth_to_meters).
    """
    raw_depth = np.asarray(raw_depth, dtype=np.float32)
    provisional_metric_depth = np.asarray(provisional_metric_depth, dtype=np.float32)
    cam_c2w = np.asarray(cam_c2w, dtype=np.float32)
    body = np.asarray(body, dtype=np.float32)
    count = min(raw_depth.shape[0], cam_c2w.shape[0], body.shape[0])
    if count < 1:
        return 0.0, 1.0
    raw_depth = raw_depth[:count]
    provisional_metric_depth = provisional_metric_depth[:count]
    cam_c2w = cam_c2w[:count]
    body = body[:count]

    height, width = raw_depth.shape[1:]
    if focal_xy is None:
        focal_xy = (0.5 * float(max(height, width)),) * 2
    if principal_xy is None:
        principal_xy = (0.5 * (width - 1), 0.5 * (height - 1))
    fx, fy = map(float, focal_xy)
    cx, cy = map(float, principal_xy)
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )
    x_dir, y_dir = (u - cx) / fx, (v - cy) / fy

    raw_values, vertical_factors, camera_z, provisional_world_z = [], [], [], []
    for frame_i in range(count):
        row_z = cam_c2w[frame_i, 2, :3]
        translation_z = float(cam_c2w[frame_i, 2, 3])
        factor = row_z[0] * x_dir + row_z[1] * y_dir + row_z[2]
        world_z = factor * provisional_metric_depth[frame_i] + translation_z
        raw_values.append(raw_depth[frame_i].reshape(-1))
        vertical_factors.append(factor.reshape(-1))
        camera_z.append(np.full(factor.size, translation_z, dtype=np.float32))
        provisional_world_z.append(world_z.reshape(-1))

    raw_values = np.concatenate(raw_values)
    vertical_factors = np.concatenate(vertical_factors)
    camera_z = np.concatenate(camera_z)
    provisional_world_z = np.concatenate(provisional_world_z)
    valid = (
        np.isfinite(raw_values)
        & np.isfinite(vertical_factors)
        & np.isfinite(camera_z)
        & np.isfinite(provisional_world_z)
    )
    raw_values = raw_values[valid]
    vertical_factors = vertical_factors[valid]
    camera_z = camera_z[valid]
    provisional_world_z = provisional_world_z[valid]

    feet = body[:, [10, 11], :].reshape(-1, 3)
    foot_z = feet[np.all(np.isfinite(feet), axis=1), 2]
    if raw_values.size < 2 or foot_z.size == 0:
        return 0.0, 1.0
    floor_z_reference = float(np.median(foot_z - 0.2))
    floor_count = max(1, int(float(floor_fraction) * provisional_world_z.size))
    selected = np.argpartition(provisional_world_z, floor_count - 1)[:floor_count]
    k_floor = vertical_factors[selected]
    raw_floor = raw_values[selected]
    z_floor = camera_z[selected]
    matrix = np.stack((k_floor * raw_floor, k_floor), axis=1)
    target = np.full(k_floor.shape, floor_z_reference, dtype=np.float32) - z_floor
    try:
        params, residuals, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 1.0
    scale, bias = float(params[0]), float(params[1])
    if not np.isfinite(scale) or not np.isfinite(bias):
        return 0.0, 1.0
    if debug:
        residual = float(np.sqrt(residuals[0])) if residuals.size else 0.0
        print(
            "[floor_fitting] "
            f"samples={floor_count}/{provisional_world_z.size}, "
            f"floor_z={floor_z_reference:.6f}, rank={rank}, "
            f"residual_l2={residual:.6f}, a={scale:.10f}, b={bias:.10f}"
        )
    return scale, bias


def depth_colors(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    if not np.any(finite):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[finite], (2.0, 98.0))
    t = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return np.stack((255 * t, 255 * (1.0 - np.abs(2.0 * t - 1.0)), 255 * (1.0 - t)), axis=-1).astype(np.uint8)


def unproject(
    depth: np.ndarray,
    pose: np.ndarray,
    stride: int,
    rgb: np.ndarray | None = None,
    filter_low_m: float = 0.0,
    filter_high_m: float = float("inf"),
    focal_xy: tuple[float, float] | None = None,
    principal_xy: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rays = normalized_rays(
        *depth.shape, focal_xy=focal_xy, principal_xy=principal_xy
    )[::stride, ::stride]
    sampled = depth[::stride, ::stride]
    points_cam = rays * sampled[..., None]
    points_world = points_cam.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
    if rgb is None:
        colors = depth_colors(sampled).reshape(-1, 3)
    else:
        if rgb.shape != (*depth.shape, 3):
            raise ValueError(f"RGB/depth shape mismatch: rgb={rgb.shape}, depth={depth.shape}")
        colors = rgb[::stride, ::stride].reshape(-1, 3)
    lower = max(float(filter_low_m), 0.0)
    upper = max(float(filter_high_m), lower + 1e-6)
    sampled_flat = sampled.reshape(-1)
    valid = (
        np.all(np.isfinite(points_world), axis=1)
        & np.isfinite(sampled_flat)
        & (sampled_flat >= lower)
        & (sampled_flat <= upper)
    )
    return points_world[valid].astype(np.float32), colors[valid]


def list_sequences(root: Path) -> tuple[str, ...]:
    """List all sequence stems under root.

    Both layouts written by demo_infer.py are covered: prediction arrays flat
    in root itself, and one <root>/<video_stem>/ subfolder per clip.
    """
    stems: set[str] = set()
    if root.is_dir():
        for cam_path in root.glob("*_tok_cam.npy"):
            stems.add(parse_stem(cam_path.name))
        for sub in root.iterdir():
            if sub.is_dir():
                for cam_path in sub.glob("*_tok_cam.npy"):
                    stems.add(parse_stem(cam_path.name))
    return tuple(sorted(stems))


def resolve_sequence_dir(root: Path, stem: str) -> Path:
    """Return the folder that actually holds the stem's prediction arrays."""
    for candidate in (root, root / stem):
        if (candidate / f"{stem}_tok_cam.npy").is_file():
            return candidate
    return root


def discover_stem(root: Path, filename: str | None) -> str:
    """Resolve the sequence stem, or auto-detect it from *_tok_cam.npy files."""
    if filename:
        return parse_stem(filename)
    sequences = list_sequences(root)
    if not sequences:
        raise FileNotFoundError(
            f"No *_tok_cam.npy found in {root} or its subfolders; pass --filename"
        )
    if len(sequences) > 1:
        print(
            f"{root} holds {len(sequences)} sequences; showing '{sequences[0]}' "
            "(switch via --filename or the Dataset dropdown)"
        )
    return sequences[0]


def resolve_inputs(
    seq_dir: Path,
    stem: str,
    rgb_video_path: Path | None,
) -> tuple[Path, Path, Path, Path, Path, bool]:
    """Resolve the prediction arrays and detect the depth convention.

    Returns ``(cam, body, depth, gaze, rgb, is_relative_depth)``. The 512x512
    metric-depth clips write ``*_tok_depth_512.npy`` (16 fps); the 256x256
    relative-depth clips write ``*_tok_depth.npy`` (8 fps).
    """
    cam_path = seq_dir / f"{stem}_tok_cam.npy"
    body_path = seq_dir / f"{stem}_tok_body.npy"
    gaze_path = seq_dir / f"{stem}_tok_gaze.npy"
    depth_512_path = seq_dir / f"{stem}_tok_depth_512.npy"
    depth_rel_path = seq_dir / f"{stem}_tok_depth.npy"
    if depth_512_path.is_file():
        depth_path, is_relative_depth = depth_512_path, False
    elif depth_rel_path.is_file():
        depth_path, is_relative_depth = depth_rel_path, True
    else:
        raise FileNotFoundError(
            f"Required input not found: {depth_512_path} (512x512 metric depth) "
            f"or {depth_rel_path} (256x256 relative depth)"
        )
    if rgb_video_path is None:
        rgb_candidates = [
            seq_dir / f"{stem}.mp4",
            seq_dir.parent / f"{stem}.mp4",
        ]
        rgb_video_path = next((path for path in rgb_candidates if path.is_file()), rgb_candidates[0])
    for path in (cam_path, body_path, depth_path, gaze_path, rgb_video_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required input not found: {path}"
                + (" (pass --video for the RGB clip)" if path == rgb_video_path else "")
            )
    return cam_path, body_path, depth_path, gaze_path, rgb_video_path, is_relative_depth


def resolve_gt_sidecars(seq_dir: Path, stem: str) -> tuple[Path | None, Path | None]:
    """Optional GT intrinsic / camera-pose sidecars copied by demo_infer.py."""
    stems = [stem]
    if stem.endswith("_rgb512"):
        stems.append(stem[: -len("_rgb512")])
    def first_file(*suffixes: str) -> Path | None:
        candidates = (
            seq_dir / f"{name}{suffix}" for name in stems for suffix in suffixes
        )
        return next((path for path in candidates if path.is_file()), None)
    gt_intrinsics_path = first_file("_gt_intrinsic.npy", "_gt_intri.npy", "_intrinsic.npy")
    gt_camera_path = first_file("_gt_cam.npy", "_first_frame_cam.npy")
    return gt_intrinsics_path, gt_camera_path


def main(args: argparse.Namespace) -> None:
    root = Path(args.prediction_dir)
    filename = args.filename
    if args.video is not None:
        if not args.video.is_file():
            raise FileNotFoundError(
                f"--video not found: {args.video}. Pass the actual path of the RGB "
                "clip that was fed to demo_infer.py (e.g. ../itw_trim/IMG_9824_00.mp4); "
                "it is only used to color the point cloud."
            )
        if filename is None:
            filename = args.video.name
    stem = discover_stem(root, filename)
    # demo_infer.py groups each clip's outputs under <output_dir>/<video_stem>/;
    # accept both that layout and arrays placed flat in the prediction dir.
    seq_dir = resolve_sequence_dir(root, stem)
    if seq_dir != root:
        print(f"Loading predictions from {seq_dir}")
    cam_path, body_path, depth_path, gaze_path, rgb_path, is_relative_depth = resolve_inputs(
        seq_dir, stem, args.video
    )
    gt_intrinsics_path, gt_camera_path = resolve_gt_sidecars(seq_dir, stem)

    # With GT camera sidecars we can reproduce the reference alignment
    # (tmp/vis_nymeria_new_0515.py): the canonical camera prediction is lifted
    # into the GT world frame and the camera-space body joints are mapped into
    # that same frame, so the two actually move together. Without GT there is no
    # way to recover that frame, so the original canonical handling is kept.
    gt_cam_w2c = None
    if gt_camera_path is not None:
        gt_cam_w2c = np.load(gt_camera_path).astype(np.float32)
        if gt_cam_w2c.ndim != 3 or gt_cam_w2c.shape[1:] != (4, 4):
            raise ValueError(
                f"Expected GT camera poses [T,4,4] in {gt_camera_path}, got {gt_cam_w2c.shape}"
            )

    body_raw = body_21_to_22(np.load(body_path))
    if gt_cam_w2c is not None:
        gt_first_c2w = np.linalg.inv(gt_cam_w2c.astype(np.float64))[0]
        cam_predicted = recon_cam_to_world(np.load(cam_path), gt_first_c2w)
        body_predicted = recover_points_to_world(body_raw, gt_cam_w2c)
        # Recenter on the GT first camera so the scene sits around the origin.
        world_offset = gt_first_c2w[:3, 3].astype(np.float32)
        cam_predicted[:, :3, 3] -= world_offset
        body_predicted = body_predicted - world_offset[None, None, :]
        print(f"Aligned camera and body to the GT world frame ({gt_camera_path.name})")
    else:
        cam_predicted = camera_to_body_coordinates(decode_camera_9d(np.load(cam_path)))
        body_predicted = body_raw

    cam_predicted = cam_predicted[: args.max_frames]
    body_predicted = body_predicted[: args.max_frames]
    frame_count = min(cam_predicted.shape[0], body_predicted.shape[0], args.max_frames)
    cam_predicted = cam_predicted[:frame_count]
    body_predicted = body_predicted[:frame_count]
    predicted_up, predicted_floor_height = estimate_up_and_floor(body_predicted)
    raw_depth = load_raw_depth(depth_path)
    # 256x256 relative depth decodes with the inverse-depth convention and plays
    # at 8 fps; 512x512 metric depth keeps the linear 0-15 m decode at 16 fps.
    # The provisional decode is only used to rank the floor candidates during the
    # affine floor fit; the fitted depth itself is still depth = a * raw + b.
    if is_relative_depth:
        clip_fps = CLIP_FPS_RELATIVE
        floor_fraction = 0.05
        original_metric_depth = decode_relative_depth_to_meters(raw_depth)
    else:
        clip_fps = CLIP_FPS_METRIC
        floor_fraction = 0.1
        original_metric_depth = decode_original_metric_depth(raw_depth)
    print(
        f"Depth input: {'256x256 relative' if is_relative_depth else '512x512 metric'} "
        f"({raw_depth.shape[2]}x{raw_depth.shape[1]}, clip {clip_fps:g} fps, "
        f"{raw_depth.shape[0]} frames)"
    )
    rgb_frames = load_rgb_video(
        rgb_path, original_metric_depth.shape[1], original_metric_depth.shape[2]
    )

    gaze_uv_norm = np.load(gaze_path).astype(np.float32)
    if gaze_uv_norm.ndim != 2 or gaze_uv_norm.shape[1] != 2:
        raise ValueError(f"Expected normalized gaze [T,2], got {gaze_uv_norm.shape}")
    gaze_uv_norm = np.clip(gaze_uv_norm, 0.0, 1.0)

    # GT sidecars (written by demo_infer.py) unlock the GT visualization modes.
    gt_focal_xy = gt_principal_xy = None
    if gt_intrinsics_path is not None:
        gt_focal_xy, gt_principal_xy = load_scaled_gt_intrinsics(
            gt_intrinsics_path,
            original_metric_depth.shape[1],
            original_metric_depth.shape[2],
        )
        print(
            f"GT intrinsics ({gt_intrinsics_path.name}): "
            f"focal={gt_focal_xy}, principal={gt_principal_xy}"
        )
    cam_gt_pose = body_gt_pose = gt_up = gt_floor_height = None
    if gt_cam_w2c is not None:
        # cam_predicted / body_predicted were already anchored to the GT first
        # camera when they were loaded, so the GT mode reuses them directly and
        # only adds the GT intrinsics on top. Re-anchoring here would be a
        # no-op rigid transform and could not fix a frame mismatch anyway.
        cam_gt_pose, body_gt_pose = cam_predicted, body_predicted
        gt_up, gt_floor_height = estimate_up_and_floor(body_gt_pose)

    visualization_modes = list(BASE_VISUALIZATION_MODES)
    if gt_focal_xy is not None and cam_gt_pose is not None:
        visualization_modes.append(MODE_GT_POSE_INTRINSIC)

    geocalib_cache: dict[str, object] = {}

    def get_geocalib_state() -> tuple[np.ndarray, tuple[float, float]]:
        if geocalib_cache:
            return geocalib_cache["cam"], geocalib_cache["focal_xy"]
        gravity_camera, geocalib_focal_xy, uncertainties = run_geocalib_first_frame(
            rgb_frames[0], args.geocalib_device, args.geocalib_weights
        )
        geocalib_cam, correction, correction_angle_deg = apply_first_frame_gravity_calibration(
            cam_predicted, gravity_camera, predicted_up
        )
        geocalib_cache.update(cam=geocalib_cam, focal_xy=geocalib_focal_xy)
        uncertainty_text = ", ".join(f"{key}={value:.5g}" for key, value in uncertainties.items())
        print(
            "GeoCalib first-frame calibration: "
            f"gravity_camera={np.round(gravity_camera, 5).tolist()}, "
            f"focal_xy_px={np.round(geocalib_focal_xy, 3).tolist()}, "
            f"pose_correction={correction_angle_deg:.3f} deg"
        )
        print(f"GeoCalib uncertainties: {uncertainty_text or 'not reported'}")
        return geocalib_cam, geocalib_focal_xy

    def state_for_mode(
        mode: str,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[float, float] | None,
        tuple[float, float] | None,
        np.ndarray,
        float,
    ]:
        """Return (cam, body, focal_xy, principal_xy, up, floor_height)."""
        if mode == MODE_PREDICTION:
            return cam_predicted.copy(), body_predicted, None, None, predicted_up, predicted_floor_height
        if mode == MODE_GEOCALIB:
            geocalib_cam, geocalib_focal = get_geocalib_state()
            return geocalib_cam.copy(), body_predicted, geocalib_focal, None, predicted_up, predicted_floor_height
        if mode == MODE_GT_POSE_INTRINSIC:
            return cam_gt_pose.copy(), body_gt_pose, gt_focal_xy, gt_principal_xy, gt_up, gt_floor_height
        raise ValueError(f"Unknown visualization mode: {mode}")

    calibration_to_mode = {
        "none": MODE_PREDICTION,
        "geocalib-first": MODE_GEOCALIB,
        "gt-first-pose": MODE_GT_POSE_INTRINSIC,
    }
    if args.camera_calibration == "auto":
        # Preference order: GT cam/intrinsic -> GeoCalib -> raw prediction.
        if MODE_GT_POSE_INTRINSIC in visualization_modes:
            initial_mode = MODE_GT_POSE_INTRINSIC
            print("Auto calibration: GT sidecars found, using GT first pose + GT intrinsic.")
        else:
            initial_mode = MODE_GEOCALIB
            print("Auto calibration: no GT sidecars, trying GeoCalib first-frame.")
    else:
        initial_mode = calibration_to_mode[args.camera_calibration]
        if initial_mode not in visualization_modes:
            raise FileNotFoundError(
                f"--camera-calibration {args.camera_calibration} needs the GT sidecars "
                f"(*_gt_intrinsic.npy / *_gt_cam.npy) next to the predictions in {seq_dir}"
            )
    try:
        cam, body, focal_xy, principal_xy, up, floor_height = state_for_mode(initial_mode)
    except Exception as exc:  # GeoCalib is optional; only 'auto' may fall back.
        if args.camera_calibration != "auto" or initial_mode != MODE_GEOCALIB:
            raise
        print(f"Auto calibration: GeoCalib unavailable ({exc}); falling back to the raw prediction.")
        initial_mode = MODE_PREDICTION
        cam, body, focal_xy, principal_xy, up, floor_height = state_for_mode(initial_mode)
    motion_for_depth = motion_indices_for_clip(frame_count, raw_depth.shape[0], clip_fps)
    floor_fit_cache: dict[str, tuple[float, float]] = {}

    def floor_fitted_depth_for_mode(mode: str) -> np.ndarray:
        cached = floor_fit_cache.get(mode)
        if cached is None:
            scale, bias = fit_depth_to_predicted_floor(
                raw_depth,
                original_metric_depth,
                cam[motion_for_depth],
                body[motion_for_depth],
                focal_xy,
                principal_xy,
                floor_fraction=floor_fraction,
                debug=True,
            )
            floor_fit_cache[mode] = (scale, bias)
        else:
            scale, bias = cached
        fitted = np.maximum(scale * raw_depth + bias, 1e-4).astype(np.float32)
        print(
            f"Floor fit [{mode}]: depth_m={scale:.10f}*raw+{bias:.10f}; "
            f"range={float(fitted.min()):.4f}-{float(fitted.max()):.4f} m"
        )
        return fitted

    depth = floor_fitted_depth_for_mode(initial_mode)
    if is_relative_depth:
        print(
            "Raw decode (floor fitting off) uses the inverse-depth convention "
            f"({RELATIVE_DEPTH_DECODE_MIN_MM:g}-{RELATIVE_DEPTH_DECODE_MAX_MM:g} mm range)"
        )
    else:
        print(
            f"Original predicted metric depth remains available in 0-{METRIC_DEPTH_MAX_M:g} m"
        )
    print(
        f"Inputs: cam={cam_path.name}, body={body_path.name}, "
        f"depth={depth_path.name}, gaze={gaze_path.name}, RGB={rgb_path.name}"
    )
    print(f"Initial visualization mode: {initial_mode}")

    # Motion (body/cam/gaze) plays at 30 fps; RGB / depth frames are held at
    # their native clip rate (8 or 16 fps), so each clip frame spans several
    # motion timesteps.
    depth_for_motion = clip_indices_for_motion(depth.shape[0], frame_count, clip_fps)
    rgb_for_motion = clip_indices_for_motion(rgb_frames.shape[0], frame_count, clip_fps)
    gaze_for_motion = sequence_indices(gaze_uv_norm.shape[0], frame_count)

    def stacked_video_frame(active: int) -> np.ndarray:
        """GT RGB (top) stacked above the colorized predicted depth (bottom)."""
        rgb = np.ascontiguousarray(rgb_frames[rgb_for_motion[active]], dtype=np.uint8)
        depth_rgb = depth_colors(depth[depth_for_motion[active]])
        if depth_rgb.shape[1] != rgb.shape[1]:
            depth_rgb = cv2.resize(
                depth_rgb, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        separator = np.full((4, rgb.shape[1], 3), 255, dtype=np.uint8)
        return np.concatenate([rgb, separator, depth_rgb], axis=0)

    server = viser.ViserServer()
    if args.share:
        server.request_share_url()

    # Video panel first so the GT RGB / predicted depth clips sit at the top.
    with server.gui.add_folder("Video (GT RGB / pred depth)"):
        show_video = server.gui.add_checkbox("Show video panel", True)
        video_image = server.gui.add_image(
            stacked_video_frame(0),
            label="GT RGB (top) / pred depth (bottom)",
        )

    available_sequences = list_sequences(root)
    if stem not in available_sequences:
        available_sequences = tuple(sorted((*available_sequences, stem)))

    with server.gui.add_folder("Dataset"):
        sequence_dropdown = server.gui.add_dropdown(
            "Sequence",
            options=available_sequences,
            initial_value=stem,
        )
        load_selected = server.gui.add_button("Load selected")

    @load_selected.on_click
    def _(_) -> None:
        selected_stem = str(sequence_dropdown.value)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--prediction_dir", str(args.prediction_dir),
            "--filename", selected_stem,
            "--max-frames", str(args.max_frames),
            "--point-stride", str(args.point_stride),
            "--point-size", str(args.point_size),
            "--camera-calibration", args.camera_calibration,
            "--geocalib-device", args.geocalib_device,
            "--geocalib-weights", args.geocalib_weights,
        ]
        if args.video is not None and selected_stem == stem:
            cmd.extend(("--video", str(args.video)))
        if args.share:
            cmd.append("--share")
        print(f"Reloading with sequence={selected_stem}")
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, cmd)

    with server.gui.add_folder("Camera / reprojection"):
        visualization_mode = server.gui.add_dropdown(
            "Setting",
            options=tuple(visualization_modes),
            initial_value=initial_mode,
        )

    with server.gui.add_folder("Playback"):
        timestep = server.gui.add_slider("Timestep", min=0, max=frame_count - 1, step=1, initial_value=0)
        playing = server.gui.add_checkbox("Playing", True)
        fps = server.gui.add_slider("FPS", min=1.0, max=60.0, step=1.0, initial_value=MOTION_FPS)
        point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.10, step=0.001, initial_value=args.point_size
        )
        keep_all_clouds = server.gui.add_checkbox("Keep all point clouds", False)
        save_all_clouds = server.gui.add_button("Save all point clouds")
    with server.gui.add_folder("Visibility"):
        show_depth = server.gui.add_checkbox("Pred depth cloud", True)
        show_body = server.gui.add_checkbox("Pred body", True)
        show_cam = server.gui.add_checkbox("Pred camera", True)
        show_gaze = server.gui.add_checkbox("Pred gaze", True)
        gaze_line_width = server.gui.add_slider(
            "Gaze line width", min=1.0, max=10.0, step=0.5, initial_value=3.0
        )
    with server.gui.add_folder("Depth filter"):
        use_floor_fitting_depth = server.gui.add_checkbox(
            "Use floor fitting depth", True
        )
        filter_low = server.gui.add_slider(
            "Filter low (m)", min=0.0, max=10.0, step=0.01, initial_value=1.25
        )
        filter_high = server.gui.add_slider(
            "Filter high (m)", min=0.1, max=METRIC_DEPTH_MAX_M, step=0.01, initial_value=10.0
        )

    # Keep one dynamic set of scene nodes. This preserves every 512x512 point
    # without retaining every frame's copy in the server.
    initial_points, initial_colors = unproject(
        depth[depth_for_motion[0]],
        cam[0],
        args.point_stride,
        rgb_frames[rgb_for_motion[0]],
        filter_low.value,
        filter_high.value,
        focal_xy,
        principal_xy,
    )
    cloud_node = server.scene.add_point_cloud(
        "/predictions/depth",
        points=initial_points,
        colors=initial_colors,
        point_size=point_size.value,
        point_shape="rounded",
    )
    body_joint_node = server.scene.add_point_cloud(
        "/predictions/body/joints",
        points=body[0],
        colors=(20, 90, 255),
        point_size=0.03,
        point_shape="rounded",
    )
    body_bone_node = ConeSkeleton(server, "/predictions/body/bones", body[0], radius=0.02)
    camera_node = server.scene.add_camera_frustum(
        "/predictions/camera",
        fov=(
            np.deg2rad(NORMALIZED_FOV_DEG)
            if focal_xy is None
            else 2.0 * np.arctan(depth.shape[1] / (2.0 * focal_xy[1]))
        ),
        aspect=depth.shape[2] / depth.shape[1],
        scale=0.10,
        color=(255, 70, 70),
        wxyz=tf.SO3.from_matrix(cam[0, :3, :3]).wxyz,
        position=cam[0, :3, 3],
    )

    def gaze_world_point(frame_i: int) -> np.ndarray:
        depth_i = depth_for_motion[frame_i]
        gaze_i = gaze_for_motion[frame_i]
        u = float(gaze_uv_norm[gaze_i, 0] * (depth.shape[2] - 1))
        v = float(gaze_uv_norm[gaze_i, 1] * (depth.shape[1] - 1))
        u_i = int(np.clip(np.rint(u), 0, depth.shape[2] - 1))
        v_i = int(np.clip(np.rint(v), 0, depth.shape[1] - 1))
        active_focal = focal_xy or (
            0.5 * float(max(depth.shape[1], depth.shape[2])),
        ) * 2
        active_principal = principal_xy or (
            0.5 * (depth.shape[2] - 1),
            0.5 * (depth.shape[1] - 1),
        )
        ray = np.array(
            [
                (u_i - active_principal[0]) / active_focal[0],
                (v_i - active_principal[1]) / active_focal[1],
                1.0,
            ],
            dtype=np.float32,
        )
        point_camera = ray * float(depth[depth_i, v_i, u_i])
        return (point_camera @ cam[frame_i, :3, :3].T + cam[frame_i, :3, 3]).astype(np.float32)

    initial_gaze_world = gaze_world_point(0)
    gaze_point_node = server.scene.add_point_cloud(
        "/predictions/gaze/point",
        points=initial_gaze_world[None],
        colors=(255, 255, 0),
        point_size=0.06,
        point_shape="circle",
    )
    gaze_line_node = server.scene.add_line_segments(
        "/predictions/gaze/ray",
        points=np.asarray([[cam[0, :3, 3], initial_gaze_world]], dtype=np.float32),
        colors=(255, 255, 0),
        line_width=gaze_line_width.value,
    )
    retained_cloud_nodes = []

    def update_scene() -> None:
        active = int(timestep.value)
        points, colors = unproject(
            depth[depth_for_motion[active]],
            cam[active],
            args.point_stride,
            rgb_frames[rgb_for_motion[active]],
            filter_low.value,
            filter_high.value,
            focal_xy,
            principal_xy,
        )
        gaze_world = gaze_world_point(active)
        cloud_node.points = points
        cloud_node.colors = colors
        body_joint_node.points = body[active]
        body_bone_node.update(body[active])
        camera_node.wxyz = tf.SO3.from_matrix(cam[active, :3, :3]).wxyz
        camera_node.position = cam[active, :3, 3]
        gaze_point_node.points = gaze_world[None]
        gaze_line_node.points = np.asarray(
            [[cam[active, :3, 3], gaze_world]], dtype=np.float32
        )
        cloud_node.visible = show_depth.value and not retained_cloud_nodes
        for node in retained_cloud_nodes:
            node.visible = show_depth.value
        body_joint_node.visible = show_body.value
        body_bone_node.visible = show_body.value
        camera_node.visible = show_cam.value
        gaze_point_node.visible = show_gaze.value
        gaze_line_node.visible = show_gaze.value
        video_image.visible = show_video.value
        if show_video.value:
            video_image.image = stacked_video_frame(active)

    def refresh_retained_clouds() -> None:
        for frame_i, node in enumerate(retained_cloud_nodes):
            points, colors = unproject(
                depth[depth_for_motion[frame_i]],
                cam[frame_i],
                args.point_stride,
                rgb_frames[rgb_for_motion[frame_i]],
                filter_low.value,
                filter_high.value,
                focal_xy,
                principal_xy,
            )
            node.points = points
            node.colors = colors

    def clear_retained_clouds() -> None:
        for node in retained_cloud_nodes:
            node.remove()
        retained_cloud_nodes.clear()
        if keep_all_clouds.value:
            keep_all_clouds.value = False

    retain_busy = {"flag": False}

    def retain_all_clouds() -> None:
        """Create one retained point cloud per frame so all of them stay visible."""
        if retained_cloud_nodes or retain_busy["flag"]:
            return
        retain_busy["flag"] = True
        keep_all_clouds.disabled = True
        print(f"Creating {frame_count} retained point clouds at stride={args.point_stride} ...")
        try:
            for frame_i in range(frame_count):
                points, colors = unproject(
                    depth[depth_for_motion[frame_i]],
                    cam[frame_i],
                    args.point_stride,
                    rgb_frames[rgb_for_motion[frame_i]],
                    filter_low.value,
                    filter_high.value,
                    focal_xy,
                    principal_xy,
                )
                retained_cloud_nodes.append(
                    server.scene.add_point_cloud(
                        f"/all_point_clouds/t{frame_i}",
                        points=points,
                        colors=colors,
                        point_size=point_size.value,
                        point_shape="rounded",
                        visible=show_depth.value,
                    )
                )
            cloud_node.visible = False
        finally:
            keep_all_clouds.disabled = False
            retain_busy["flag"] = False
        print(f"Retained all {frame_count} point clouds")

    @keep_all_clouds.on_update
    def _(_) -> None:
        if not keep_all_clouds.value:
            clear_retained_clouds()
            update_scene()
            print("Removed retained point clouds; showing only the active frame")
            return
        retain_all_clouds()

    @save_all_clouds.on_click
    def _(_) -> None:
        if not keep_all_clouds.value:
            keep_all_clouds.value = True  # its on_update handler retains the clouds
        retain_all_clouds()

    @visualization_mode.on_update
    def _(_) -> None:
        nonlocal cam, body, focal_xy, principal_xy, up, floor_height, depth
        visualization_mode.disabled = True
        try:
            cam, body, focal_xy, principal_xy, up, floor_height = state_for_mode(
                str(visualization_mode.value)
            )
            depth = (
                floor_fitted_depth_for_mode(str(visualization_mode.value))
                if use_floor_fitting_depth.value
                else original_metric_depth
            )
            camera_node.fov = (
                np.deg2rad(NORMALIZED_FOV_DEG)
                if focal_xy is None
                else 2.0 * np.arctan(depth.shape[1] / (2.0 * focal_xy[1]))
            )
            refresh_retained_clouds()
            update_scene()
            print(f"Visualization setting: {visualization_mode.value}")
        finally:
            visualization_mode.disabled = False

    @timestep.on_update
    def _(_) -> None:
        update_scene()

    @point_size.on_update
    def _(_) -> None:
        cloud_node.point_size = point_size.value
        for node in retained_cloud_nodes:
            node.point_size = point_size.value

    @filter_low.on_update
    def _(_) -> None:
        refresh_retained_clouds()
        update_scene()

    @filter_high.on_update
    def _(_) -> None:
        refresh_retained_clouds()
        update_scene()

    @use_floor_fitting_depth.on_update
    def _(_) -> None:
        nonlocal depth
        # A depth-source change returns to the single dynamic cloud instead of
        # silently rebuilding every retained frame with the new depth.
        clear_retained_clouds()
        depth = (
            floor_fitted_depth_for_mode(str(visualization_mode.value))
            if use_floor_fitting_depth.value
            else original_metric_depth
        )
        refresh_retained_clouds()
        update_scene()
        source = "floor-fitted" if use_floor_fitting_depth.value else "original metric"
        print(f"Depth source: {source} [{visualization_mode.value}]")

    @show_depth.on_update
    def _(_) -> None:
        update_scene()

    @show_body.on_update
    def _(_) -> None:
        update_scene()

    @show_cam.on_update
    def _(_) -> None:
        update_scene()

    @show_gaze.on_update
    def _(_) -> None:
        update_scene()

    @show_video.on_update
    def _(_) -> None:
        update_scene()

    @gaze_line_width.on_update
    def _(_) -> None:
        gaze_line_node.line_width = gaze_line_width.value

    update_scene()
    while True:
        if playing.value:
            timestep.value = (int(timestep.value) + 1) % frame_count
        time.sleep(1.0 / max(float(fps.value), 1.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction_dir",
        type=Path,
        default=Path("./demo_output"),
        help="demo_infer.py output root: either one sequence's arrays "
        "(*_tok_{cam,body,gaze,depth_512}.npy) directly inside, or one "
        "<video_stem>/ subfolder per clip. All sequences found are offered "
        "in the Dataset dropdown.",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Sequence stem or any of its filenames. Defaults to the first "
        "sequence found in the prediction dir (switchable in the sidebar).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="RGB clip used for demo_infer (colors only; default: <stem>.mp4 "
        "next to the predictions).",
    )
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument(
        "--point-stride",
        type=int,
        default=1,
        help="Point sampling stride. Default 1 keeps the full 512x512 prediction.",
    )
    parser.add_argument("--point-size", type=float, default=0.02)
    parser.add_argument(
        "--camera-calibration",
        choices=("auto", "none", "geocalib-first", "gt-first-pose"),
        default="auto",
        help=(
            "Initial camera calibration mode (default: auto, which picks the "
            "best available: GT first pose + GT intrinsic if the sidecars are "
            "present, else GeoCalib first-frame, else the raw prediction). "
            "'geocalib-first' corrects frame-0 roll/pitch and uses the "
            "estimated focal length while preserving relative motion; 'none' "
            "shows the raw prediction; 'gt-first-pose' anchors the trajectory "
            "to the GT first camera pose and reprojects with the GT "
            "intrinsics. The GT mode needs the *_gt_intrinsic.npy / "
            "*_gt_cam.npy sidecars copied by demo_infer.py."
        ),
    )
    parser.add_argument(
        "--geocalib-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device used for the single GeoCalib inference.",
    )
    parser.add_argument(
        "--geocalib-weights",
        default="pinhole",
        help="GeoCalib weights preset ('pinhole') or checkpoint path.",
    )
    parser.add_argument("--share", action="store_true")
    main(parser.parse_args())
