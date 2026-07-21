import numpy as np
import torch
from torch.utils.data import Dataset
import os
# Data root, override with the REVIV_DATA_ROOT env var (default: ./example_data).
DATA_ROOT = os.environ.get("REVIV_DATA_ROOT", "./example_data")


class MotionDataset(Dataset):
    """Mixed body/hand motion dataset packed into a shared 62-joint layout.

    Layout:
    - body joints: [0:22)
    - left hand joints without the wrist root: [22:42)
    - right hand joints without the wrist root: [42:62)

    Shared roots:
    - left hand joint 0  -> body joint 20
    - right hand joint 0 -> body joint 21

    Each returned sample has shape [T, 62, 4]:
    - xyz coordinates in the first 3 channels
    - a per-joint validity mask in the last channel
    """

    BODY_JOINTS = 21
    BODY_WITH_HAND_ROOTS = 22
    HAND_JOINTS = 21
    TOTAL_JOINTS = 62

    LEFT_ROOT_BODY_IDX = 20
    RIGHT_ROOT_BODY_IDX = 21
    LEFT_HAND_START = BODY_WITH_HAND_ROOTS
    RIGHT_HAND_START = BODY_WITH_HAND_ROOTS + (HAND_JOINTS - 1)

    BODY_TRAIN_PATH = f"{DATA_ROOT}/body/train_21_ego_251031.npy"
    BODY_VAL_PATH = f"{DATA_ROOT}/body/val_21_ego_251031.npy"
    BODY_MEAN_PATH = f"{DATA_ROOT}/body/body_mean_21_ego_251031.npy"
    BODY_STD_PATH = f"{DATA_ROOT}/body/body_std_21_ego_251031.npy"

    LHAND_TRAIN_PATH = f"{DATA_ROOT}/hand/hand_train_60_lhand.npy"
    LHAND_VAL_PATH = f"{DATA_ROOT}/hand/hand_val_60_lhand.npy"
    LHAND_MEAN_PATH = f"{DATA_ROOT}/hand/hand_mean_60_lhand.npy"
    LHAND_STD_PATH = f"{DATA_ROOT}/hand/hand_std_60_lhand.npy"

    RHAND_TRAIN_PATH = f"{DATA_ROOT}/hand/hand_train_60_rhand.npy"
    RHAND_VAL_PATH = f"{DATA_ROOT}/hand/hand_val_60_rhand.npy"
    RHAND_MEAN_PATH = f"{DATA_ROOT}/hand/hand_mean_60_rhand.npy"
    RHAND_STD_PATH = f"{DATA_ROOT}/hand/hand_std_60_rhand.npy"

    def __init__(self, mode="train", clip_len=60, args=None):
        if mode not in {"train", "val"}:
            raise ValueError(f"Unsupported mode for MotionDataset: {mode}")
        if clip_len != 60:
            raise ValueError(
                f"MotionDataset expects clip_len=60 because it reuses precomputed 60-frame clips, got {clip_len}."
            )

        self.mode = mode
        self.clip_len = clip_len
        self.args = args

        body_path = self.BODY_TRAIN_PATH if mode == "train" else self.BODY_VAL_PATH
        lhand_path = self.LHAND_TRAIN_PATH if mode == "train" else self.LHAND_VAL_PATH
        rhand_path = self.RHAND_TRAIN_PATH if mode == "train" else self.RHAND_VAL_PATH

        # Memory-map the large arrays so we can mix all sources without loading
        # the full body and hand corpora into RAM at once.
        self.body_samples = np.load(body_path, mmap_mode="r")
        self.lhand_samples = np.load(lhand_path, mmap_mode="r")
        self.rhand_samples = np.load(rhand_path, mmap_mode="r")

        self.body_mean = np.load(self.BODY_MEAN_PATH).astype(np.float32)
        self.body_std = np.load(self.BODY_STD_PATH).astype(np.float32)
        self.lhand_mean = np.load(self.LHAND_MEAN_PATH).astype(np.float32)
        self.lhand_std = np.load(self.LHAND_STD_PATH).astype(np.float32)
        self.rhand_mean = np.load(self.RHAND_MEAN_PATH).astype(np.float32)
        self.rhand_std = np.load(self.RHAND_STD_PATH).astype(np.float32)

        self.mean = np.zeros((self.TOTAL_JOINTS, 3), dtype=np.float32)
        self.std = np.ones((self.TOTAL_JOINTS, 3), dtype=np.float32)

        self.mean[:self.BODY_JOINTS] = self.body_mean
        self.std[:self.BODY_JOINTS] = self.body_std
        self.mean[self.RIGHT_ROOT_BODY_IDX] = self.rhand_mean[0]
        self.std[self.RIGHT_ROOT_BODY_IDX] = self.rhand_std[0]
        self.mean[self.LEFT_HAND_START:self.RIGHT_HAND_START] = self.lhand_mean[1:]
        self.std[self.LEFT_HAND_START:self.RIGHT_HAND_START] = self.lhand_std[1:]
        self.mean[self.RIGHT_HAND_START:] = self.rhand_mean[1:]
        self.std[self.RIGHT_HAND_START:] = self.rhand_std[1:]

        self.body_len = len(self.body_samples)
        self.lhand_len = len(self.lhand_samples)
        self.rhand_len = len(self.rhand_samples)

        print(
            f"Loaded MotionDataset({mode}) with "
            f"{self.body_len} body clips, {self.lhand_len} left-hand clips, {self.rhand_len} right-hand clips."
        )

    def _normalize_and_mask(self, coords, mask):
        coords = (coords - self.mean[None, :, :]) / self.std[None, :, :]
        coords = np.where(mask > 0, coords, 0.0).astype(np.float32)
        packed = np.concatenate((coords, mask.astype(np.float32)), axis=-1)
        return torch.from_numpy(packed).float()

    def _pack_body_sample(self, sample):
        coords = np.zeros((self.clip_len, self.TOTAL_JOINTS, 3), dtype=np.float32)
        mask = np.zeros((self.clip_len, self.TOTAL_JOINTS, 1), dtype=np.float32)

        coords[:, :self.BODY_JOINTS] = np.asarray(sample, dtype=np.float32)
        mask[:, :self.BODY_JOINTS, 0] = 1.0

        return self._normalize_and_mask(coords, mask)

    def _pack_hand_sample(self, sample, is_right):
        coords = np.zeros((self.clip_len, self.TOTAL_JOINTS, 3), dtype=np.float32)
        mask = np.zeros((self.clip_len, self.TOTAL_JOINTS, 1), dtype=np.float32)

        sample = np.asarray(sample, dtype=np.float32)
        valid = (~np.isnan(sample).any(axis=-1)).astype(np.float32)
        sample = np.nan_to_num(sample, nan=0.0)

        root_body_idx = self.RIGHT_ROOT_BODY_IDX if is_right else self.LEFT_ROOT_BODY_IDX
        hand_start = self.RIGHT_HAND_START if is_right else self.LEFT_HAND_START

        coords[:, root_body_idx] = sample[:, 0]
        mask[:, root_body_idx, 0] = valid[:, 0]
        coords[:, hand_start:hand_start + (self.HAND_JOINTS - 1)] = sample[:, 1:]
        mask[:, hand_start:hand_start + (self.HAND_JOINTS - 1), 0] = valid[:, 1:]

        return self._normalize_and_mask(coords, mask)

    def __getitem__(self, index):
        if index < self.body_len:
            return self._pack_body_sample(self.body_samples[index])

        index -= self.body_len
        if index < self.lhand_len:
            return self._pack_hand_sample(self.lhand_samples[index], is_right=False)

        index -= self.lhand_len
        return self._pack_hand_sample(self.rhand_samples[index], is_right=True)

    def __len__(self):
        return self.body_len + self.lhand_len + self.rhand_len
