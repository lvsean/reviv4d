# Training the main ReViV model

`run_training_reviv.py` trains the main any-to-any multimodal transformer on
**pre-tokenized** egocentric clips. It is a 4M-style masked-modeling
encoder–decoder: at each step a random subset of input tokens (across all
modalities) is encoded and the model predicts a random subset of target tokens,
letting a single model perform any-to-any generation (e.g. RGB → depth / body /
hands / cam / gaze, depth → RGB, etc.).

Make sure the tokenizers ([README_TOKENIZATION.md](README_TOKENIZATION.md)) have
been used to produce the token shards ([README_DATA.md](README_DATA.md)) before
training the main model.

## Model

The default architecture is registered under the timm-style name
`reviv_base_12e_12d_swiglu_nobias` (`reviv/models/reviv_model.py`):

- 12 encoder + 12 decoder layers, `dim = 768`, 12 heads
- SwiGLU MLP (SiLU + gated), fully bias-free, RMS/LayerNorm without bias
- ~400M parameters including the per-modality embedding layers

Other sizes (`reviv_tiny/small/base/large/xlarge_...`, `gelu` and
`swiglu_qknorm` variants) are registered in the same file. The builder name is
what the config's `model:` field references.

## Modalities

Derived automatically from the data config (union of every dataset's
`in_domains` / `out_domains`):

- **Inputs**: `rgb@256`, `rgb512` (raw video, encoder-only) plus the token
  streams `tok_rgb`, `tok_depth`, `tok_rgb_512`, `tok_depth_512`, `tok_cam`,
  `tok_gaze`, `tok_body`, `tok_lhand`, `tok_rhand`.
- **Outputs**: the same token streams (raw video has no decoder).

Per-modality vocab sizes and token counts are in `reviv/data/modality_info.py`
(summarized in [README_DATA.md](README_DATA.md)).

## Configs

```
cfgs/default/reviv/
├── models/main/reviv_base_500b_2048.yaml         # top-level training config
├── data/ego/main/mix_mod9_all2all_2048.yaml      # datasets, shard paths, sampling weights
└── alphas_mixture/main/                          # Dirichlet masking mixtures
    ├── mix_mod6_all2all_uni_vit.yaml
    └── mix_mod9_all2all_uni_vit_body_rgb512_txt.yaml
```

Key fields of `reviv_base_500b_2048.yaml`:

```yaml
model: reviv_base_12e_12d_swiglu_nobias
dtype: bfloat16
num_input_tokens: 2048
num_target_tokens: 2048
total_tokens: 500          # training length in billions of tokens
warmup_tokens: 10          # billions
blr: 0.0001                # lr = blr * global_batch_size / 256
batch_size: 4              # per GPU
data_config: "cfgs/default/reviv/data/ego/main/mix_mod9_all2all_2048.yaml"
save_ckpt_freq: 1
output_dir: './output/auto'
```

Edit the `data_path` fields in the data config to point at your token root, and
`output_dir` to your checkpoint location.

## Launching

Single machine (uses all visible GPUs, or pass a count as first argument):

```bash
cd ReViV
bash train_scripts/main_train.sh
```

Multi-node — run the same script once per node, pointing every node at rank-0:

```bash
NNODES=4 NODE_RANK=0 MASTER_ADDR=host0 bash train_scripts/main_train.sh 4  # on host0
NNODES=4 NODE_RANK=1 MASTER_ADDR=host0 bash train_scripts/main_train.sh 4  # on host1
# ...
```

The script boils down to:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 \
  torchrun --nproc_per_node <#gpus> --nnodes <#nodes> \
  run_training_reviv.py \
  --config cfgs/default/reviv/models/main/reviv_base_500b_2048.yaml
```

The reference model was trained on 256 GPUs; when scaling down, the learning
rate is scaled automatically from the effective global batch size. For a quick
local smoke test, a single node with `--nproc_per_node <#gpus>` works.

## Checkpointing & resume

- Checkpoints are written every epoch to `output_dir` as
  `checkpoint-<epoch>.pth`, plus a final `checkpoint-final.pth`. Each holds
  `model`, `optimizer`, `scaler`, `epoch`, `args`.
- **Auto-resume is on by default** (`--auto_resume`, default `True`): on restart
  the script picks the highest-numbered `checkpoint-*.pth` in `output_dir` and
  continues. Pass `--resume <path>` to resume a specific checkpoint, or
  `--finetune <path>` to warm-start weights only (`strict=False`, positional
  embeddings stripped).

## Logging

Set `log_wandb: True` and `wandb_project` / `wandb_entity` in the config to log
to Weights & Biases. If your machine sits behind an HTTP proxy, make sure
`api.wandb.ai` is reachable (e.g. via `https_proxy` / `no_proxy`).
