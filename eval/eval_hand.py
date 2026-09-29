#!/usr/bin/env python3
"""ReViV egocentric hand reconstruction benchmark: a folder of RGB clips + a
folder of GT hand joints + a released checkpoint set -> GA / RA / PA-MPJPE.

Lives in eval/ of https://github.com/lvsean/reviv4d; run from the repo root:

    python eval/eval_hand.py --video_dir /path/to/clips \
                        --gt_dir    /path/to/gt_joints \
                        --ckpt_root /path/to/reviv_checkpoints/reviv_500b \
                        --output_dir ./eval_hand_output

It prints the three numbers of the paper's hand table (mm), one row per
dataset folder, and writes per-clip metrics, predictions and timing.

INPUTS
  --video_dir   folder of 2 s clips (`.mp4/.mov/...`, any resolution/fps; the
                first 2 s are used, resampled exactly like demo_hand.py: 16
                frames @ 8 fps, 256 x 256 for the 256-pathway checkpoints).
                With --recursive, each sub-folder is reported as its own
                dataset row.
  --gt_dir      GT hand joints in CAMERA space, metres, 21 joints per hand
                (wrist first, i.e. the ReViV / EgoM2P joint order).  For a clip
                `<stem>.mp4` the file is looked up as
                  1. <gt_dir>/[<dataset>/]<stem>.npz            keys lhand, rhand [60, 21, 3]
                     (the demo_hand.py `gt_joints` layout; extra frames are cut to 60)
                  2. <gt_dir>/[<dataset>/]<base>.npz  when <stem> = "<base>-<j>":
                     keys lhand, rhand [120, 21, 3]; the clip is frames 60j .. 60j+59
                     (the EgoM2P label layout: one 4 s label file = two 2 s clips).
                Optional boolean `lhand_visible` / `rhand_visible` [T] mask the
                frames that enter the metrics (HoloAssist has them).
                Clips without GT are still predicted but not scored.
  --ckpt_root   a released checkpoint set: reviv_main.pth, reviv_tok_lhand.pth,
                reviv_tok_rhand.pth and norm_stats/{l,r}hand_{mean,std}.npy
                (or $REVIV_CKPT_ROOT). --cosmos_dir as in demo_hand.py.

PROTOCOL (what produced the numbers in the paper; `--mode paper`, default)
  conditioning  tok_rgb (5120 Cosmos DV4x8x8 codes) + rgb@256 (the raw clip)
  generation    tok_rhand then tok_lhand in one chained schedule, MaskGIT,
                2 cosine steps per hand, temperature 0.01, CFG 3.0, top_p 0.8, seed 0
  metrics       per clip and hand, on the 60 label frames (30 fps):
                  RA-MPJPE  wrist (joint 0) subtracted per frame
                  GA-MPJPE  ONE similarity transform (rotation, translation,
                            scale; Umeyama) fitted over all visible joints of
                            the clip, then mean joint error
                  PA-MPJPE  the same similarity fit per frame
                lrhand = mean of the two hands; the table is the mean over
                clips in mm, columns GA, RA, PA.  Exactly the reference scorer
                (EgoM2P detokenize_script/eval_mpjpe_h2o.py), including its
                convention that a hand with no visible frame in a clip scores
                0.0 and is still averaged; the `visible_only` numbers in
                summary.json drop such clips instead.
  `--mode single` is the fast path: one encoder pass per clip, both hands
  sampled independently in one step (temperature 0.01, top_p 0.8, no CFG).
  On the released reviv_500b it is never worse than `paper` and ~4x faster.

OUTPUT  <output_dir>/
  per_clip.jsonl               one line per clip: key, dataset, lhand/rhand RA/GA/PA (m), n visible frames
  summary.json                 per dataset + overall: GA/RA/PA (mm) mean/std, l/r split, visible_only, timing
  pred/<dataset>/<stem>_tok_{lhand,rhand}.npy   predicted joints [60, 21, 3] (demo_hand.py format,
                               so demo_vis_hand.py can overlay them)
  Re-running resumes: clips already in per_clip.jsonl are skipped.

SPEED  Timing is printed as s/clip and FPS (60 label frames per clip) for the
generation alone and end to end (video decode + Cosmos encoding + generation +
detokenisation + metrics).  By default the model's attention is swapped for
torch.nn.functional.scaled_dot_product_attention at inference (numerically
equivalent: identical metrics), which is what makes batch 8 fit a 24 GB card;
`--no_sdpa` restores the repo's original attention (batch 1 only: 2.2 s/clip
and 14.8 GB on a 4090 vs 0.18 s/clip and 6.4 GB).  Measured on one RTX 4090
with reviv_500b: paper 0.18 s/clip (~330 FPS generation, ~250 FPS end to end),
single 0.046 s/clip (~1300 FPS generation, ~800 FPS end to end).
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # eval/ lives one level below the repo root
sys.path.insert(0, REPO)

from demo_hand import (TARGET_DOMAINS, TOKENS_PER_TARGET, VIDEO_EXTS,  # noqa: E402
                       build_schedule, collect_videos, select_hand_pathway)
from demo_infer import (CKPT_ROOT, DEVICE, PATHWAYS, generation_context,  # noqa: E402
                        load_clip, load_main_model, load_motion_tokenizer)
from cosmos_tokenizer.video_lib import CausalVideoTokenizer  # noqa: E402
from reviv.data.modality_info import MODALITY_INFO  # noqa: E402
from reviv.models import reviv_utils  # noqa: E402
from reviv.models.generate import (GenerationSampler, init_empty_target_modality,  # noqa: E402
                                   init_full_input_modality)
from tokenizers import Tokenizer  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_grad_enabled(False)

FRAMES = 60                      # label frames per 2 s clip (30 fps)
N_T, N_J = 30, 7                 # hand token grid
TEMP, TOP_P = 0.01, 0.8
SIDES = {'tok_lhand': 'lhand', 'tok_rhand': 'rhand'}
PAPER_TABLE = {                  # camera-ready table rows (700b checkpoint), for reference
    'holoassist': (23.5, 29.7, 10.5), 'hot3d_aria': (38.2, 48.1, 13.6), 'hot3d': (38.2, 48.1, 13.6),
    'arctic': (35.1, 33.0, 13.7), 'taco': (20.3, 25.2, 9.4)}


# --------------------------------------------------------------------------- #
# metrics -- the reference scorer's definitions, vectorised over frames
# --------------------------------------------------------------------------- #
def _umeyama(p, g):
    """Similarity transform (scale, rot, trans) mapping p onto g; p, g: [..., N, 3]."""
    mu_p, mu_g = p.mean(-2, keepdims=True), g.mean(-2, keepdims=True)
    pc, gc = p - mu_p, g - mu_g
    cov = np.swapaxes(pc, -1, -2) @ gc
    u, _, vt = np.linalg.svd(cov)
    v = np.swapaxes(vt, -1, -2)
    z = np.broadcast_to(np.eye(3), cov.shape).copy()
    z[..., 2, 2] = np.where(np.linalg.det(v @ np.swapaxes(u, -1, -2)) < 0, -1.0, 1.0)
    rot = v @ z @ np.swapaxes(u, -1, -2)
    scale = np.trace(rot @ cov, axis1=-2, axis2=-1) / (np.sum(pc ** 2, axis=(-1, -2)) + 1e-8)
    trans = mu_g - scale[..., None, None] * (mu_p @ np.swapaxes(rot, -1, -2))
    return scale, rot, trans


def hand_metrics(pred, gt, vis, root=0):
    """pred, gt: [T, 21, 3] metres; vis: [T] bool. Returns (RA, GA, PA) in metres.

    Matches eval_mpjpe_h2o.py: a hand with no visible frame scores 0.0 on all
    three; GA needs >= 3 visible joints in total, PA >= 3 per frame (always true
    with 21 joints)."""
    if not vis.any():
        return 0.0, 0.0, 0.0
    p, g = pred[vis], gt[vis]
    ra = float(np.linalg.norm((p - p[:, root:root + 1]) - (g - g[:, root:root + 1]), axis=2).mean())
    s, r, t = _umeyama(p.reshape(-1, 3), g.reshape(-1, 3))
    ga = float(np.linalg.norm(s * (p @ r.T) + t - g, axis=2).mean())
    s, r, t = _umeyama(p, g)
    pa = float(np.linalg.norm(s[:, None, None] * (p @ np.swapaxes(r, -1, -2)) + t - g, axis=2).mean(1).mean())
    return ra, ga, pa


# --------------------------------------------------------------------------- #
# GT lookup
# --------------------------------------------------------------------------- #
def find_gt(gt_dir, dataset, stem, video_dir_name=''):
    """-> (path, frame offset) or (None, None). See the module docstring.

    Tried in order: <gt_dir>/<dataset>/, <gt_dir>/<basename of --video_dir>/ (the
    demo_hand.py `../gt_joints/<dataset>/<stem>.npz` convention), <gt_dir>/."""
    dirs = []
    for d in (dataset, video_dir_name, ''):
        cand = os.path.join(gt_dir, d) if d else gt_dir
        if cand not in dirs:
            dirs.append(cand)
    for d in dirs:
        p = os.path.join(d, stem + '.npz')
        if os.path.isfile(p):
            return p, 0
    if '-' in stem and stem.rsplit('-', 1)[1].isdigit():
        base, j = stem.rsplit('-', 1)
        for d in dirs:
            p = os.path.join(d, base + '.npz')
            if os.path.isfile(p):
                return p, FRAMES * int(j)
    return None, None


def load_gt(path, offset):
    g = np.load(path)
    out = {}
    for side in SIDES.values():
        if side not in g.files:
            return None
        arr = np.asarray(g[side], np.float64)[offset:offset + FRAMES]
        vis = np.ones(len(arr), bool)
        if f'{side}_visible' in g.files:
            vis = np.asarray(g[f'{side}_visible'], bool)[offset:offset + FRAMES][:len(arr)]
        out[side], out[f'{side}_vis'] = arr, vis
    return out


# --------------------------------------------------------------------------- #
# optional SDPA attention (inference only, numerically equivalent)
# --------------------------------------------------------------------------- #
def enable_sdpa():
    def bias(mask, dtype):
        if mask is None or not bool(mask.any()):
            return None
        return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill_(mask, -torch.finfo(dtype).max)

    orig_attn, orig_xattn = reviv_utils.Attention.forward, reviv_utils.CrossAttention.forward

    def attn_forward(self, x, mask=None):
        if self.allow_zero_attn:
            return orig_attn(self, x, mask)
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        m = None if mask is None else mask.unsqueeze(1)                    # [B, 1, (1|N), N]
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias(m, q.dtype))
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))

    def xattn_forward(self, x, context, mask=None):
        if self.allow_zero_attn:
            return orig_xattn(self, x, context, mask)
        B, N, C = x.shape
        M = context.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k, v = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        m = None if mask is None else mask.unsqueeze(1)                    # [B, 1, N, M]
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias(m, q.dtype))
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))

    reviv_utils.Attention.forward = attn_forward
    reviv_utils.CrossAttention.forward = xattn_forward


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class Clips(Dataset):
    def __init__(self, items, pw, start_sec):
        self.items, self.pw, self.start_sec = items, pw, start_sec

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        clip = load_clip(self.items[i]['path'], start_sec=self.start_sec, num_frames=self.pw['num_frames'],
                         clip_fps=self.pw['clip_fps'], frame_size=self.pw['frame_size'])   # [1, T, H, W, 3] uint8
        return i, torch.from_numpy(np.ascontiguousarray(clip[0]))


def build_sample(tokens, clip, pw, eos_id):
    """tokens [B, 5120] long, clip [B, T, H, W, 3] float in [-1, 1] (or None)."""
    B = tokens.shape[0]
    sample = {}
    for dom in pw['cond_domains']:
        t = clip if dom.startswith('rgb') else tokens
        n = MODALITY_INFO[dom]['max_tokens']
        sample[dom] = {'tensor': t, 'input_mask': torch.zeros(B, n, dtype=torch.bool, device=DEVICE),
                       'target_mask': torch.ones(B, n, dtype=torch.bool, device=DEVICE)}
    for dom, n in zip(TARGET_DOMAINS, TOKENS_PER_TARGET):
        sample = init_empty_target_modality(sample, MODALITY_INFO, dom, B, n, DEVICE)
    for dom in pw['cond_domains']:
        sample = init_full_input_modality(sample, MODALITY_INFO, dom, DEVICE, eos_id=eos_id)
    return sample


def gen_paper(sampler, sample, schedule, text_tok, seed):
    out = sampler.generate(sample, schedule, text_tokenizer=text_tok, verbose=False, seed=seed, top_p=TOP_P, top_k=0.0)
    return {t: out[t]['tensor'] for t in TARGET_DOMAINS}


def gen_single(sampler, sample, seed):
    """One encoder pass; every hand position sampled in one step from the shared context."""
    model = sampler.model
    enc = {m: model.encoder_embeddings[m](d) for m, d in sample.items() if m in model.encoder_embeddings}
    enc_tokens, enc_emb, enc_mask, _ = sampler.forward_mask_encoder_generation(enc)
    context = model.decoder_proj_context(model.forward_encoder(enc_tokens + enc_emb, enc_mask)) + enc_emb
    out = {}
    for tm in TARGET_DOMAINS:
        dec = {tm: model.decoder_embeddings[tm].forward_embed(sample[tm])}
        dt, de, _, dmm, pos = sampler.forward_mask_decoder_maskgit(dec, tm, seed=seed)
        y = model.forward_decoder(dt + de, context, enc_mask, None)
        B, N, _ = y.shape
        logits = model.forward_logits(y, dec, dmm)[tm].reshape(B, N, -1).float()
        samp, _ = sampler.sample_tokens_batched(logits, temperature=TEMP, top_k=0.0, top_p=TOP_P)
        t = sample[tm]['tensor'].clone()
        t.scatter_(1, pos, samp)
        out[tm] = t
    return out


def decode_hands(toks, stats, ids):
    out = {}
    for tm, side in SIDES.items():
        rec = toks[tm].decode_tokens(ids[tm].to(DEVICE).reshape(-1, N_T, N_J)).detach().float().cpu().numpy()
        out[side] = (rec * stats[f'{side}_std'] + stats[f'{side}_mean']).reshape(-1, FRAMES, 21, 3).astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video_dir', required=True, help='folder of clips (' + ', '.join(VIDEO_EXTS) + ')')
    p.add_argument('--gt_dir', default=None, help='folder of GT hand-joint npz files (see docstring); '
                   'default: <video_dir>/../gt_joints as in demo_hand.py')
    p.add_argument('--ckpt_root', default=CKPT_ROOT, help='released checkpoint set folder (or $REVIV_CKPT_ROOT)')
    p.add_argument('--cosmos_dir', default=None, help='Cosmos encoder.jit folder; default: the pathway\'s')
    p.add_argument('--output_dir', default='./eval_hand_output')
    p.add_argument('--recursive', action='store_true', help='one dataset row per sub-folder of --video_dir')
    p.add_argument('--mode', default='paper', choices=['paper', 'single'])
    p.add_argument('--batch_size', type=int, default=8, help='clips per forward (8 fits a 24 GB card with SDPA)')
    p.add_argument('--no_sdpa', dest='sdpa', action='store_false',
                   help="use the repo's original attention instead of scaled_dot_product_attention "
                        '(same numbers, ~10x slower and 2.5x more memory at batch 1; then use --batch_size 1)')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--amp_dtype', default='bf16', choices=['none', 'bf16', 'fp16'])
    p.add_argument('--start_sec', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_clips', type=int, default=0, help='debug: only the first N clips')
    p.add_argument('--no_pred', action='store_true', help='do not write the .npy predictions')
    p.add_argument('--text_tokenizer',
                   default=os.path.join(REPO, 'reviv/utils/tokenizer/trained/text_tokenizer_reviv_wordpiece_30k.json'))
    return p.parse_args()


def main():
    args = parse_args()
    if args.sdpa:
        enable_sdpa()
    elif args.batch_size > 1:
        print('WARNING: --no_sdpa with --batch_size > 1 materialises the full attention matrix '
              '(~5 GB per clip on the 256 pathway); expect OOM on 24 GB cards -- use --batch_size 1.')
    os.makedirs(args.output_dir, exist_ok=True)
    video_dir = os.path.abspath(args.video_dir)
    gt_dir = args.gt_dir or os.path.join(os.path.dirname(video_dir), 'gt_joints')

    # clips -> (dataset, stem, path, gt)
    if args.recursive:      # like demo_hand.collect_videos, but following symlinked dataset folders
        paths = [os.path.join(root, f) for root, _, files in os.walk(video_dir, followlinks=True)
                 for f in files if f.lower().endswith(VIDEO_EXTS)]
    else:
        paths = collect_videos(video_dir)
    if not paths:
        raise SystemExit(f'no videos ({", ".join(VIDEO_EXTS)}) under {video_dir}')
    items = []
    for pth in sorted(paths):
        rel = os.path.relpath(os.path.dirname(os.path.abspath(pth)), video_dir)
        dataset = '' if rel == '.' else rel.replace(os.sep, '/')
        stem = os.path.splitext(os.path.basename(pth))[0]
        gt_path, off = find_gt(gt_dir, dataset, stem, os.path.basename(video_dir))
        items.append({'dataset': dataset or os.path.basename(video_dir), 'stem': stem, 'path': pth,
                      'key': f'{dataset}/{stem}' if dataset else stem, 'gt': gt_path, 'gt_off': off})
    if args.max_clips:
        items = items[:args.max_clips]
    n_gt = sum(i['gt'] is not None for i in items)
    print(f'{len(items)} clips in {video_dir}, {n_gt} with GT under {gt_dir}', flush=True)
    if n_gt == 0:
        print('WARNING: no GT found -- predictions will be written but nothing scored')

    jsonl = os.path.join(args.output_dir, 'per_clip.jsonl')
    done = set()
    if os.path.isfile(jsonl):
        with open(jsonl) as fh:
            done = {json.loads(l)['key'] for l in fh if l.strip()}
    todo = [i for i in items if i['key'] not in done]
    print(f'{len(done)} already done, {len(todo)} to run ({args.mode}, batch {args.batch_size}, sdpa={args.sdpa})', flush=True)

    # models
    t0 = time.time()
    ckpt = lambda name: os.path.join(args.ckpt_root, name)  # noqa: E731
    text_tok = Tokenizer.from_file(args.text_tokenizer)
    model, all_domains = load_main_model(ckpt('reviv_main.pth'))
    pathway = select_hand_pathway(all_domains)
    pw = PATHWAYS[pathway]
    missing = [d for d in pw['cond_domains'] + TARGET_DOMAINS if d not in all_domains]
    if missing:
        raise SystemExit(f'checkpoint domains {all_domains} miss {missing}')
    sampler = GenerationSampler(model)
    schedule = build_schedule(pw) if args.mode == 'paper' else None
    toks = {tm: load_motion_tokenizer(ckpt(f'reviv_tok_{side}.pth')) for tm, side in SIDES.items()}
    norm_dir = os.path.join(args.ckpt_root, 'norm_stats')
    if not os.path.isdir(norm_dir):
        norm_dir = os.environ.get('REVIV_DATA_ROOT', norm_dir)
    stats = {f'{s}_{w}': np.load(os.path.join(norm_dir, f'{s}_{w}.npy')).astype(np.float32)
             for s in SIDES.values() for w in ('mean', 'std')}
    cosmos_dir = args.cosmos_dir or pw['cosmos_dir']
    if not os.path.isdir(cosmos_dir) and os.path.isdir(os.path.join(REPO, cosmos_dir)):
        cosmos_dir = os.path.join(REPO, cosmos_dir)   # default is relative to the repo root
    encoder = CausalVideoTokenizer(checkpoint_enc=os.path.join(cosmos_dir, 'encoder.jit'), device=DEVICE)
    eos_id = text_tok.token_to_id('[EOS]')
    load_s = time.time() - t0
    print(f'models loaded in {load_s:.1f}s ({pathway} pathway, conditions {pw["cond_domains"]})', flush=True)
    if not todo:
        summarise(args, items, load_s, None)
        return

    loader = DataLoader(Clips(todo, pw, args.start_sec), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=DEVICE == 'cuda')
    wall0 = time.time()
    gen_s = enc_s = 0.0
    n_done = 0
    pred_dir = os.path.join(args.output_dir, 'pred')
    with open(jsonl, 'a') as fh:
        for bi, (idx, clip_u8) in enumerate(loader):
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            t_e = time.time()
            with torch.inference_mode():                                        # Cosmos takes uint8 [B, T, H, W, 3]
                tokens = torch.as_tensor(np.asarray(encoder(clip_u8.numpy(), temporal_window=pw['num_frames'])),
                                         dtype=torch.int64).reshape(len(clip_u8), -1).to(DEVICE)
            clip = clip_u8.to(DEVICE).float().div_(127.5).sub_(1.0) if 'rgb@256' in pw['cond_domains'] else None
            sample = build_sample(tokens, clip, pw, eos_id)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            enc_s += time.time() - t_e
            t_g = time.time()
            with generation_context(args.amp_dtype):
                out = gen_paper(sampler, sample, schedule, text_tok, args.seed) if args.mode == 'paper' \
                    else gen_single(sampler, sample, args.seed)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            gen_s += time.time() - t_g
            joints = decode_hands(toks, stats, out)
            for b, i in enumerate(idx.tolist()):
                it = todo[i]
                rec = {'key': it['key'], 'dataset': it['dataset'], 'stem': it['stem'], 'gt': it['gt'] is not None}
                if it['gt'] is not None:
                    gt = load_gt(it['gt'], it['gt_off'])
                    if gt is None:
                        rec['gt'] = False
                    else:
                        for side in SIDES.values():
                            T = min(FRAMES, gt[side].shape[0])
                            ra, ga, pa = hand_metrics(joints[side][b, :T].astype(np.float64), gt[side][:T], gt[f'{side}_vis'][:T])
                            rec[side] = {'RA': ra, 'GA': ga, 'PA': pa, 'n_vis': int(gt[f'{side}_vis'][:T].sum()), 'T': int(T)}
                if not args.no_pred:
                    d = os.path.join(pred_dir, it['dataset'])
                    os.makedirs(d, exist_ok=True)
                    for side in SIDES.values():
                        np.save(os.path.join(d, f'{it["stem"]}_tok_{side}.npy'), joints[side][b])
                fh.write(json.dumps(rec) + '\n')
                n_done += 1
            if (bi + 1) % max(1, 50 // args.batch_size) == 0 or n_done == len(todo):
                el = time.time() - wall0
                print(f'  {n_done}/{len(todo)}  gen {gen_s / n_done:.3f} s/clip  encode {enc_s / n_done:.3f}  '
                      f'end-to-end {el / n_done:.3f} s/clip ({n_done * FRAMES / el:.0f} FPS)', flush=True)
            fh.flush()
    wall = time.time() - wall0
    timing = {'n_clips': n_done, 'mode': args.mode, 'batch_size': args.batch_size, 'sdpa': args.sdpa,
              'amp_dtype': args.amp_dtype, 'device': torch.cuda.get_device_name(0) if DEVICE == 'cuda' else platform.processor(),
              'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEVICE == 'cuda' else None,
              'model_load_s': round(load_s, 1), 'cosmos_encode_s': round(enc_s, 2), 'generation_s': round(gen_s, 2),
              'end_to_end_s': round(wall, 2),
              's_per_clip_generation': round(gen_s / n_done, 4), 'fps_generation': round(n_done * FRAMES / gen_s, 1),
              's_per_clip_end_to_end': round(wall / n_done, 4), 'fps_end_to_end': round(n_done * FRAMES / wall, 1)}
    summarise(args, items, load_s, timing)


def summarise(args, items, load_s, timing):
    jsonl = os.path.join(args.output_dir, 'per_clip.jsonl')
    recs = {}
    with open(jsonl) as fh:
        for l in fh:
            if l.strip():
                r = json.loads(l)
                recs[r['key']] = r
    scored = [r for r in recs.values() if r.get('gt') and 'lhand' in r]
    groups = {}
    for r in scored:
        groups.setdefault(r['dataset'], []).append(r)
    if len(groups) > 1:
        groups['ALL'] = scored

    def agg(rs):
        out = {'n': len(rs)}
        for m in ('GA', 'RA', 'PA'):
            l = np.array([r['lhand'][m] for r in rs]) * 1000
            rr = np.array([r['rhand'][m] for r in rs]) * 1000
            lr = 0.5 * (l + rr)
            out[m] = {'mean': float(lr.mean()), 'std': float(lr.std()), 'l': float(l.mean()), 'r': float(rr.mean())}
        return out

    summary = {'mode': args.mode, 'ckpt_root': os.path.abspath(args.ckpt_root), 'video_dir': os.path.abspath(args.video_dir),
               'n_clips_total': len(items), 'n_scored': len(scored), 'datasets': {}}
    print('\n' + '=' * 92)
    print(f'{"dataset":<24}{"clips":>7}   GA-MPJPE  RA-MPJPE  PA-MPJPE   (mm; paper-table row for reference)')
    print('-' * 92)
    for name, rs in groups.items():
        a = agg(rs)
        vis = [r for r in rs if r['lhand']['n_vis'] > 0 and r['rhand']['n_vis'] > 0]
        a['visible_only'] = agg(vis) if vis else None
        summary['datasets'][name] = a
        ref = PAPER_TABLE.get(name.lower())
        ref_s = f'   paper: {ref[0]:.1f} / {ref[1]:.1f} / {ref[2]:.1f}' if ref else ''
        print(f'{name:<24}{a["n"]:>7}   {a["GA"]["mean"]:8.1f}  {a["RA"]["mean"]:8.1f}  {a["PA"]["mean"]:8.1f}{ref_s}')
        if len(vis) != len(rs):
            v = a['visible_only']
            print(f'{"  visible-only":<24}{v["n"]:>7}   {v["GA"]["mean"]:8.1f}  {v["RA"]["mean"]:8.1f}  {v["PA"]["mean"]:8.1f}')
    print('=' * 92)
    if timing:
        summary['timing'] = timing
        print(f'speed ({timing["device"]}, {args.mode}, batch {args.batch_size}{", sdpa" if args.sdpa else ""}): '
              f'generation {timing["s_per_clip_generation"]} s/clip = {timing["fps_generation"]} FPS; '
              f'end-to-end {timing["s_per_clip_end_to_end"]} s/clip = {timing["fps_end_to_end"]} FPS '
              f'(model load {timing["model_load_s"]} s, peak GPU mem {timing["peak_mem_gb"]} GB)')
    if not scored:
        print('nothing scored (no GT found)')
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(f'per-clip metrics: {jsonl}\nsummary: {os.path.join(args.output_dir, "summary.json")}')


if __name__ == '__main__':
    main()
