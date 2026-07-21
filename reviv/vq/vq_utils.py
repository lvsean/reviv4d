# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import numpy as np
from torch.nn import functional as F

def compute_codebook_usage(
        all_tokens: torch.LongTensor, 
        codebook_size: int = 16_384, 
        window_size: int = 65_536) -> float:
    """Computes the codebook usage for a given set of encoded tokens, by computing the 
    percentage of unique tokens in windows of a given size. The window size should be
    chosen as batch_size * sequence_length, where batch_size is recommended to be set
    to 256, and the sequence_length is the number of tokens per image. We follow
    ViT-VQGAN's approach of using batch_size 256. (https://arxiv.org/abs/2110.04627)

    Args:
        all_tokens: A tensor of shape (n_tokens, ) containing all the encoded tokens.
        codebook_size: The size of the codebook.
        window_size: The size of the window to compute the codebook usage in.

    Returns:
        The average codebook usage.
    """
    n_full_windows = all_tokens.shape[0] // window_size
    
    percentages = []
    for i, token_window in enumerate(torch.split(all_tokens, window_size)):
        if i < n_full_windows:
            usage_perc = len(np.unique(token_window)) / codebook_size
            percentages.append(usage_perc)
        else:
            break
            
    return np.mean(percentages)

def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

def rot6d_to_rotmat(rot6d):
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:]
    b1 = normalize(a1)
    b2 = normalize(a2 - (b1 * a2).sum(-1, keepdims=True) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    mat = torch.stack([b1, b2, b3], dim=-1)
    return mat

def rotation_distance(R1, R2, eps=1e-7):
    R_diff = R1 @ R2.transpose(-2, -1)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_() # numerical stability near -1/+1
    return angle