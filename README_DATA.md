# Data

ReViV does **not** distribute any dataset. This document describes the
**expected on-disk format** so you can plug in your own processed data. There
are two stages:

1. **Motion clips** — continuous body / hand motion, used to train the
   VQ-VAE tokenizers (`run_training_vqvae.py`).
2. **Tokens** — the pre-tokenized multimodal clips consumed by the main model
   (`run_training_reviv.py`).

Everything is organized in fixed-length **clips**. A clip is a temporal window
of `num_frames = 60` frames (motion is stored at the tokenizer's native rate).

### Data root

All data paths are rooted at a single, overridable location `DATA_ROOT`, which
defaults to `./example_data`. Point it
somewhere else with the `REVIV_DATA_ROOT` environment variable — no code or
config edits needed:

```bash
REVIV_DATA_ROOT=/scratch/my_data python run_training_vqvae.py --config ...
```

Config `data_path` / `output_dir` fields are likewise plain parameters you can
edit per run.

---

## 1. Motion clips (VQ-VAE tokenizer training)

Motion is stored as dense numpy arrays of shape `[N, T, V, 3]`:

| axis | meaning |
|------|---------|
| `N`  | number of clips |
| `T`  | frames per clip (`= 60`) |
| `V`  | number of joints |
| `3`  | xyz coordinates (in a canonical / camera space) |

Missing joints are stored as `NaN`; the dataset classes replace them with `0`
after normalization and append a per-joint **validity mask** as a 4th channel,
so what the model actually receives is `[T, V, 4]` (xyz + mask).

Each `*.npy` clip array is accompanied by **per-joint mean / std sidecars**
(same `V, 3` layout) used for normalization `(x - mean) / std`.

### Body — `domain: body` (released)

Loaded by `reviv/data/body_dataset.py`. `V = 21` joints (a SMPL-style
egocentric body skeleton with the root joint dropped). This is what the
**released** `reviv_tok_body.pth` uses, and what the demos output as
`[60, 21, 3]`. At `tubelet_size [2, 3]` a 60-frame clip tokenizes to
`(60/2) x (21/3) = 210` tokens, matching `tok_body`'s `max_tokens` below.
Unlike the hand layout this variant carries **no** validity-mask channel —
`__getitem__` returns `[T, 21, 3]`.

Default files (root `${REVIV_DATA_ROOT}/body/`, default `./example_data/body/`):

```
train_21_ego_251031.npy , val_21_ego_251031.npy      # [N, 60, 21, 3]
body_mean_21_ego_251031.npy , body_std_21_ego_251031.npy
```

### Hands — `domain: lhand_joint` (left **and** right)

Loaded by `reviv/data/hand_joint_dataset.py`. `V = 21` hand joints. **Left vs
right is chosen by the `rhand` flag in the config, not by the domain string** —
both hands use `domain: lhand_joint`.

Default files (root `${REVIV_DATA_ROOT}/hand/`, default `./example_data/hand/`):

```
hand_{train,val}_60_lhand.npy         # [N, 60, 21, 3]   (rhand: False)
hand_mean_60_lhand.npy , hand_std_60_lhand.npy
hand_{train,val}_60_rhand.npy         # [N, 60, 21, 3]   (rhand: True)
hand_mean_60_rhand.npy , hand_std_60_rhand.npy
```

`__getitem__` returns `impute_nan((sample - mean) / std)` → `[T, 21, 4]`
(xyz + validity mask), which is what `HandEncoder` expects.

> All paths are rooted at `DATA_ROOT` (the `REVIV_DATA_ROOT` env var, default
> `./example_data`). `HandJointDataset` supports two file layouts via the
> optional `hand_data_source` arg: `'default'` (the filenames above) and
> `'camspace'` (an alternate `all_hand_*_camspace_withholo` layout). It defaults
> to `'default'`.

---

## 2. Tokens (main model training)

The main model does **not** read raw pixels or raw motion — it reads
**pre-tokenized** clips stored as [WebDataset](https://github.com/webdataset/webdataset)
tar shards, one modality per tar, merged by sample key. Produce these with the
tokenizers described in [README_TOKENIZATION.md](README_TOKENIZATION.md)
(motion / cam / gaze) and the Cosmos tokenizer (RGB / depth video).

### Directory layout

Tokens live under a root (`${REVIV_DATA_ROOT}/main_model_token/`, default
`./example_data/main_model_token/`) with one sub-folder
per modality, and one dataset sub-folder underneath, containing the shards:

```
main_model_token/
├── rgb_video/<dataset>/token/shard-{000000..NNNNNN}.tar   # raw RGB video frames, 256px (encoder-only input)
├── rgb_video_512/<dataset>/token/shard-*.tar              # raw RGB video frames, 512px (encoder-only input)
├── rgb/<dataset>/token/shard-*.tar                        # Cosmos RGB tokens (256px)
├── rgb_512/<dataset>/token/shard-*.tar                    # Cosmos RGB tokens (512px)
├── depth/<dataset>/token/shard-*.tar                      # Cosmos depth tokens (256px)
├── depth_512/<dataset>/token/shard-*.tar                  # Cosmos depth tokens (512px)
├── cam/<dataset>/token/shard-*.tar                        # camera-trajectory tokens
├── gaze/<dataset>/token/shard-*.tar                       # gaze tokens
├── body/<dataset>/token/shard-*.tar                       # body motion tokens (21 joints)
├── lhand/<dataset>/token/shard-*.tar                      # left-hand tokens
└── rhand/<dataset>/token/shard-*.tar                      # right-hand tokens
```

Each sample within a shard is keyed by a common id, and each modality file is a
`.npy` array of **int64 token indices** (raw video inputs are stored as image /
video frames). The loader (`reviv/data/unified_datasets.py`) brace-expands the
`[rgb,depth,cam,…]` bracket in a data path into the parallel per-modality tars
and joins them by key.

### Modality token specs

Defined in `reviv/data/modality_info.py`:

| modality       | folder        | vocab size | max tokens | tokenizer |
|----------------|---------------|-----------:|-----------:|-----------|
| `tok_body`     | `body`        |      2048  |       210  | body VQ-VAE (this repo) |
| `tok_lhand`    | `lhand`       |      1024  |       210  | hand VQ-VAE (this repo) |
| `tok_rhand`    | `rhand`       |      1024  |       210  | hand VQ-VAE (this repo) |
| `tok_cam`      | `cam`         |       256  |        30  | cam transformer VQ-VAE |
| `tok_gaze`     | `gaze`        |       512  |        30  | gaze transformer VQ-VAE |
| `tok_rgb`      | `rgb`         |     64000  |      5120  | Cosmos video (256px) |
| `tok_depth`    | `depth`       |     64000  |      5120  | Cosmos video (256px) |
| `tok_rgb_512`  | `rgb_512`     |     64000  |      5120  | Cosmos video (512px) |
| `tok_depth_512`| `depth_512`   |     64000  |      5120  | Cosmos video (512px) |
| `rgb@256`      | `rgb_video`   |     — (raw)|      4096  | raw video, encoder-only |
| `rgb512`       | `rgb_video_512`|    — (raw)|      4096  | raw video, encoder-only |

Which datasets / modalities / shard ranges are used, and the per-dataset
sampling weights, are declared in the data config
`cfgs/default/reviv/data/ego/main/mix_mod9_all2all_2048.yaml`. Edit the
`data_path` fields there to point at your token root.
