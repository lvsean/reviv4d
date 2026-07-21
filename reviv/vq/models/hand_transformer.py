import math
import warnings
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.cuda.amp import autocast
from einops import rearrange
import os
# Data root, override with the REVIV_DATA_ROOT env var (default: ./example_data).
DATA_ROOT = os.environ.get("REVIV_DATA_ROOT", "./example_data")

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.):
    """Sine-cosine positional embeddings as used in MoCo-v3
    """
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    pos_emb = rearrange(pos_emb, 'b (h w) d -> b h w d', h=h, w=w, d=embed_dim)
    return pos_emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, **kwargs):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HandEncoder(nn.Module):
    def __init__(self, *, 
                 in_channels: int = 3, 
                 num_frames: int = 120,
                 joint_nums: int = 21,
                 dim_tokens: int = 768,
                 depth: int = 12,
                 tubelet_size = [2, 1],
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 sincos_pos_emb: bool = True, 
                 learnable_pos_emb: bool = False, 
                 post_mlp: bool = True,
                 ckpt_path: Optional[str] = None,
                 **ignore_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.dim_tokens = dim_tokens
        print('*******************************')
        print('hand conv shape: ', tubelet_size)
        print('*******************************')
        self.tubelet_size = tubelet_size
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.dim_tokens, kernel_size=tubelet_size, stride=tubelet_size)
        self.position_embeddings = build_2d_sincos_posemb(h=num_frames // tubelet_size[0], w=joint_nums // tubelet_size[1], embed_dim=dim_tokens)
        self.position_embeddings = nn.Parameter(self.position_embeddings, requires_grad=learnable_pos_emb)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=dim_tokens, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.depth = depth

        if post_mlp:
            self.norm_mlp = norm_layer(dim_tokens)
            self.post_mlp = Mlp(dim_tokens, int(mlp_ratio*dim_tokens), act_layer=nn.Tanh)
        
        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)

            if isinstance(m, nn.Conv2d):
                if '.proj' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                    # TODO: this branch may never exec. Check if this is a bug
                    pass


    def _init_weights(self, m: nn.Module) -> None:
        """Weight initialization"""
        if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self) -> int:
        """Get number of transformer layers."""
        return len(self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, T, J, 3]
        Returns:
            Output tensor of shape [B, dim_tokens, N_T, N_H, N_W].
        """
        bz, time, njoint, _ = x.shape
        data = x[..., :3]
        mask = x[..., 3:]
        
        x = self.conv((data * mask).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = x + self.position_embeddings
        x = rearrange(x, 'b t j d -> b (t j) d', t=time // self.tubelet_size[0], j=njoint // self.tubelet_size[1])

        # Transformer forward pass
        x = self.blocks(x)

        if hasattr(self, 'post_mlp'):
            x = x + self.post_mlp(self.norm_mlp(x))

        # Reshape into 3D grid
        x = rearrange(x, 'b (t j) d -> b d t j', t=time // self.tubelet_size[0], j=njoint // self.tubelet_size[1])

        return x


class HandDecoder(nn.Module):
    def __init__(self, *, 
                 out_channels: int = 3, 
                 num_frames: int = 120,
                 joint_nums: int = 21,
                 dim_tokens: int = 768,
                 depth: int = 12,
                 tubelet_size = [2, 3],
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 sincos_pos_emb: bool = True, 
                 learnable_pos_emb: bool = False,
                 post_mlp: bool = True,
                 out_conv: bool = False,
                 **ignore_kwargs):
        super().__init__()

        self.out_channels = out_channels
        self.dim_tokens = dim_tokens
        self.tubelet_size = tubelet_size
        self.position_embeddings = build_2d_sincos_posemb(h=num_frames // tubelet_size[0], w=joint_nums // tubelet_size[1], embed_dim=dim_tokens)
        self.position_embeddings = nn.Parameter(self.position_embeddings, requires_grad=learnable_pos_emb)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=dim_tokens, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])

        # Tokens -> image output projection
        if post_mlp:
            self.norm_mlp = norm_layer(dim_tokens)
            self.post_mlp = Mlp(dim_tokens, int(mlp_ratio*dim_tokens), act_layer=nn.Tanh)

        self.out_proj = nn.Linear(dim_tokens, self.out_channels * self.tubelet_size[0] * self.tubelet_size[1])
        
        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)

            if isinstance(m, nn.Conv2d):
                if '.proj' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                    # TODO: this branch may never exec. Check if this is a bug
                    pass


    def _init_weights(self, m: nn.Module) -> None:
        """Weight initialization"""
        if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self) -> int:
        """Get number of transformer layers."""
        return len(self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, NT, NJ = x.shape
        x = x + self.position_embeddings.permute(0, 3, 1, 2)
        x = rearrange(x, 'b d t j -> b (t j) d')

        # Transformer forward pass
        x = self.blocks(x)

        # Project each token to (C * P_H * P_W)
        if hasattr(self, 'post_mlp'):
            x = x + self.post_mlp(self.norm_mlp(x))

        x = self.out_proj(x)
        x = rearrange(
            x, 'b (nt nj) (c pt pj) -> b (nt pt) (nj pj) c',
            nt=NT, nj=NJ, pt=self.tubelet_size[0], pj=self.tubelet_size[1], c=self.out_channels
        )
        return x


if __name__ == '__main__':
    import numpy as np
    hand_verts1 = np.load(f'{DATA_ROOT}/arctic/lhand_joints/s08-scissors_use_03-000410.npy')
    enc = HandEncoder().cuda()
    dec = HandDecoder().cuda()
    input = torch.from_numpy(hand_verts1).cuda().unsqueeze(0)
    node_feat = enc(input)
    node_pred = dec(node_feat)
    pass

