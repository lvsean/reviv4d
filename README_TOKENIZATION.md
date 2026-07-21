# Tokenization — Body & Hand Motion VQ-VAEs

This guide covers training the discrete **motion tokenizers** with
`run_training_vqvae.py`: a body VQ-VAE and one VQ-VAE each for the left and
right hand. Once trained, they turn continuous joint trajectories into the
`tok_body` / `tok_lhand` / `tok_rhand` token streams the main model consumes.

RGB / depth video are tokenized separately with the pretrained NVIDIA Cosmos
tokenizer (`cosmos_tokenizer/`); camera trajectory and gaze use the lightweight
transformer VQ-VAEs under `reviv/vq/models/` — the training entry point is the
same script, just with a different config / `--domain`.

## Model

`run_training_vqvae.py` builds a `VQVAE` (`reviv/vq/vqvae.py`) whose encoder /
decoder are selected from the config's `encoder_type` / `decoder_type` strings:

| tokenizer | encoder/decoder     | class (`reviv/vq/models/`)          | codebook | joints |
|-----------|---------------------|-------------------------------------|---------:|-------:|
| body      | `BodyTransformer`   | `BodyEncoder` / `BodyDecoder`       |    2048  |    21  |
| hand      | `HandTransformer`   | `HandEncoder` / `HandDecoder`       |    1024  |    21  |

Both are ViT-style transformers over a `[T, V]` grid, patchified with a
`tubelet_size` (`[2, 3]` in the shipped motion configs), quantized by a
lucidrains-style EMA codebook (`reviv/vq/quantizers/`). The number of body
joints is set by `body_joint_nums` (21 for the released body tokenizer); the
hand tokenizer always uses 21 joints. With `num_frames = 60` and
`tubelet_size = [2, 3]` the body tokenizer emits `(60/2) x (21/3) = 210`
tokens, matching `modality_info`'s `tok_body` `max_tokens`. Body and the two
hands are kept as three separate 21-joint token streams (`tok_body`,
`tok_lhand`, `tok_rhand`).

## Data

Motion clips of shape `[N, T=60, V, 3]` plus mean/std sidecars — see
[README_DATA.md](README_DATA.md). Point the config's `data_path` /
`eval_data_path` at your processed clips.

## How body vs left vs right is selected

- `--domain body` → `BodyDataset` (21-joint body tokenizer).
- `--domain lhand_joint` → `HandJointDataset` (hand tokenizer).
  **Left vs right is the `rhand` flag**, *not* the domain:
  `rhand: False` → left hand, `rhand: True` → right hand.

These are set in the config files, so normally you just pick the right config.

## Configs

```
cfgs/default/tokenization/vqvae/
├── body/
│   └── body_motion_60_2048_2_3.yaml       # 21-joint body, codebook 2048
│                                          #   (matches the released reviv_tok_body.pth
│                                          #    and modality_info's tok_body: 210 tokens)
└── hand/
    ├── hand_1k_60_lhand.yaml              # left hand,  codebook 1024, rhand: False
    └── hand_1k_60_rhand.yaml              # right hand, codebook 1024, rhand: True
```

Key fields (released body example):

```yaml
encoder_type: BodyTransformer
decoder_type: BodyTransformer
num_frames: 60
tubelet_size: [2, 3]
body_joint_nums: 21
codebook_size: 2048
latent_dim: 32
quantizer_type: lucid
loss_fn: mse                   # the `body` domain has no validity channel
domain: body
data_path: ".../body/train_21_ego_251031.npy"
batch_size: 64
epochs: 1000
output_dir: './output/auto'
```

Hand configs are analogous with `HandTransformer`, `codebook_size: 1024`,
`domain: lhand_joint`, `loss_fn: mse_mask_hand`, and the `rhand` flag.

## Launching

Use the shell scripts in `train_scripts/` (single machine; pass the number of
GPUs as an optional argument, default = all visible GPUs):

```bash
cd ReViV

bash train_scripts/body_tok_train.sh         # body tokenizer
bash train_scripts/hand_tok_train_lhand.sh   # left-hand tokenizer
bash train_scripts/hand_tok_train_rhand.sh   # right-hand tokenizer
```

Each script boils down to:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 \
  torchrun --standalone --nproc_per_node <#gpus> run_training_vqvae.py \
  --config cfgs/default/tokenization/vqvae/body/body_motion_60_2048_2_3.yaml
```

You can also run that command directly once the environment is active.

## Outputs

The literal `auto` in `output_dir` is replaced with the config's path (minus
`cfgs/default/`), e.g.:

```
./output/tokenization/vqvae/body/body_motion_60_2048_2_3/
├── checkpoint-0.pth, checkpoint-1.pth, ...   # every epoch (save_ckpt_freq: 1)
├── checkpoint-final.pth
└── log.txt
```

Each checkpoint is a dict with `model` (state_dict), `epoch`, `args`, `scaler`.
Use these tokenizers to encode your clips into the `tok_body` / `tok_lhand` /
`tok_rhand` shards described in [README_DATA.md](README_DATA.md).
