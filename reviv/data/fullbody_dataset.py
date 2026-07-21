import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset



# Root directory for processed clip data. Override with REVIV_DATA_ROOT env var.
DATA_ROOT = os.environ.get("REVIV_DATA_ROOT", "./example_data")


class FullBodyDataset(Dataset):
    """Load 60-joint full-body clips produced by reviv/data/load_full_body.py.

    Each returned tensor is shaped [T, 60, 4]:
    - channels 0:3 are normalized xyz coordinates
    - channel 3 is a valid-joint mask

    NaN coordinates are replaced with zero after normalization so the encoder can
    consume sparse body-only or hand-only clips without propagating NaNs.
    """

    DEFAULT_ROOT = f"{DATA_ROOT}/full_body"
    JOINTS = 60
    COORDS = 3

    def __init__(self, mode="train", clip_len=60, args=None):
        if mode not in {"train", "val", "tokenize"}:
            raise ValueError(f"Unsupported mode {mode} for FullBodyDataset")
        if args is None:
            raise ValueError("FullBodyDataset requires args with data_path/eval_data_path")

        self.mode = mode
        self.clip_len = clip_len
        self.args = args

        data_path = self._resolve_data_path(args, mode, clip_len)
        self.data_path = data_path
        self.mean_path = self._sidecar_path(data_path, "mean", args)
        self.std_path = self._sidecar_path(data_path, "std", args)
        self.meta_path = self._sidecar_path(data_path, "meta", args)

        self.names = None
        if mode == "tokenize" and self._is_pickle_path(data_path):
            self.samples, self.names = self._load_tokenize_pickles(args, clip_len)
            stats_base_path = self._resolve_data_path(args, "train", clip_len)
            self.mean_path = self._sidecar_path(stats_base_path, "mean", args)
            self.std_path = self._sidecar_path(stats_base_path, "std", args)
        else:
            self.samples = np.load(data_path, mmap_mode="r")
            if mode == "tokenize" and os.path.exists(self.meta_path):
                with open(self.meta_path, "rb") as f:
                    meta = pickle.load(f)
                self.names = meta.get("name", None)

        if self.samples.ndim != 4 or self.samples.shape[1:] != (clip_len, self.JOINTS, self.COORDS):
            raise ValueError(
                f"Expected data shape (N, {clip_len}, {self.JOINTS}, {self.COORDS}), "
                f"got {self.samples.shape} from {data_path}"
            )

        self.mean = self._load_stats(self.mean_path, fill_value=0.0)
        self.std = self._load_stats(self.std_path, fill_value=1.0)
        self.std = np.where((self.std > 0) & np.isfinite(self.std), self.std, 1.0).astype(np.float32)

        print(
            f"Loaded FullBodyDataset({mode}) from {data_path}: "
            f"{self.samples.shape}, mean {self.mean.shape}, std {self.std.shape}"
        )

    @classmethod
    def _resolve_data_path(cls, args, mode, clip_len):
        if mode == "tokenize":
            tokenize_path = getattr(args, "tokenize_path", None)
            if tokenize_path:
                return cls._resolve_tokenize_path(tokenize_path, clip_len)

        eval_data_path = getattr(args, "eval_data_path", None)
        path = eval_data_path if mode == "val" and eval_data_path else getattr(args, "data_path", None)
        if path is None:
            split = "val" if mode == "val" else "train"
            path = f"full_body_{split}_60.npy"

        if os.path.isdir(path):
            split = "val" if mode == "val" else "train"
            return os.path.join(path, f"full_body_{split}_60.npy")
        if os.path.isabs(path):
            return path

        candidates = [
            os.path.join(cls.DEFAULT_ROOT, path),
            os.path.abspath(path),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    @classmethod
    def _resolve_tokenize_path(cls, tokenize_path, clip_len):
        if os.path.isdir(tokenize_path):
            return os.path.join(tokenize_path, f"full_body_train_{clip_len}_tokenize.pkl")
        if "," in tokenize_path:
            return tokenize_path
        if os.path.isabs(tokenize_path):
            return tokenize_path
        candidates = [
            os.path.join(cls.DEFAULT_ROOT, tokenize_path),
            os.path.abspath(tokenize_path),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    @staticmethod
    def _is_pickle_path(path):
        if "," in path:
            return True
        return path.endswith(".pkl") or path.endswith(".pickle")

    def _sidecar_path(self, data_path, suffix, args):
        explicit = getattr(args, f"fullbody_{suffix}_path", None)
        if explicit:
            return explicit
        ext = ".pkl" if suffix == "meta" else ".npy"
        stem, _ = os.path.splitext(data_path)
        if suffix in {"mean", "std"} and stem.endswith("_tokenize"):
            stem = stem[: -len("_tokenize")]
        return f"{stem}_{suffix}{ext}"

    def _load_stats(self, path, fill_value):
        if os.path.exists(path):
            arr = np.load(path).astype(np.float32)
            return arr.reshape(-1, self.COORDS)[-self.JOINTS:]
        print(f"Warning: {path} not found. Using fill value {fill_value}.")
        return np.full((self.JOINTS, self.COORDS), fill_value, dtype=np.float32)

    def _load_tokenize_pickles(self, args, clip_len):
        paths = self._tokenize_pickle_paths(args, clip_len)
        data = []
        names = []
        for path in paths:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            data.append(np.asarray(payload["data"], dtype=np.float32))
            names.extend([str(name) for name in payload.get("name", [])])

        samples = np.concatenate(data, axis=0) if len(data) > 1 else data[0]
        if len(names) != len(samples):
            names = [f"fullbody_{index:08d}" for index in range(len(samples))]
        return samples, names

    def _tokenize_pickle_paths(self, args, clip_len):
        path = getattr(args, "tokenize_path", None) or self.data_path
        raw_paths = [p.strip() for p in path.split(",") if p.strip()]
        if len(raw_paths) > 1:
            return raw_paths

        path = raw_paths[0]
        if os.path.isdir(path):
            paths = [
                os.path.join(path, f"full_body_train_{clip_len}_tokenize.pkl"),
                os.path.join(path, f"full_body_val_{clip_len}_tokenize.pkl"),
            ]
            existing = [candidate for candidate in paths if os.path.exists(candidate)]
            return existing if existing else paths[:1]
        return [path]

    def _pack_sample(self, sample):
        sample = np.asarray(sample, dtype=np.float32)
        valid = np.isfinite(sample).all(axis=-1, keepdims=True).astype(np.float32)
        normalized = (sample - self.mean[None, :, :]) / self.std[None, :, :]
        normalized = np.where(np.isfinite(normalized), normalized, 0.0)
        normalized = normalized * valid
        return torch.from_numpy(np.concatenate((normalized, valid), axis=-1)).float()

    def __getitem__(self, index):
        packed = self._pack_sample(self.samples[index])
        if self.mode == "tokenize":
            if self.names is not None and index < len(self.names):
                name = str(self.names[index])
            else:
                name = f"fullbody_{index:08d}"
            return {"x": packed, "name": name}
        return packed

    def __len__(self):
        return len(self.samples)
