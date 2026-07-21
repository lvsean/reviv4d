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

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def build_1d_sincos_posemb(max_len, embed_dim=1024, temperature=10000.):
    """Sine-cosine positional embeddings from MoCo-v3, adapted back to 1d

    Returns positional embedding of shape (1, N, D)
    """
    arange = torch.arange(max_len, dtype=torch.float32) # Shape (N,)
    assert embed_dim % 2 == 0, 'Embed dimension must be divisible by 2 for 1D sin-cos position embedding'
    pos_dim = embed_dim // 2
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim # Shape (D/2,)
    omega = 1. / (temperature ** omega)
    out = torch.einsum('n,d->nd', [arange, omega]) # Outer product, shape (N, D/2)
    pos_emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1).unsqueeze(0) # Shape (1, N, D)
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


class GazeEncoder(nn.Module):
    def __init__(self, *, 
                 in_channels: int = 2, 
                 num_frames: int = 80,
                 dim_tokens: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 temporal_compress: Optional[int] = None,
                 sincos_pos_emb: bool = True, 
                 learnable_pos_emb: bool = False, 
                 post_mlp: bool = True,
                 ckpt_path: Optional[str] = None,
                 **ignore_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.dim_tokens = dim_tokens
        self.temporal_compress = 4 if temporal_compress is None else temporal_compress
        self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.dim_tokens, kernel_size=self.temporal_compress, stride=self.temporal_compress)
        self.position_embeddings = build_1d_sincos_posemb(num_frames // self.temporal_compress, embed_dim=dim_tokens)
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
        if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv2d, nn.Conv1d)):
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
            x: Input tensor of shape [B, T, 2]
        Returns:
            Output tensor of shape [B, dim_tokens, N_T, N_H, N_W].
        """
        bz, time, _ = x.shape
        data = x[:, :, :2]
        mask = x[:, :, 2:]
        # mask out invalid data
        emb = self.conv((data * mask).permute(0, 2, 1)).permute(0, 2, 1)
        emb = emb + self.position_embeddings

        # Transformer forward pass
        emb = self.blocks(emb)

        if hasattr(self, 'post_mlp'):
            emb = emb + self.post_mlp(self.norm_mlp(emb))

        emb = rearrange(emb, 'b t d -> b d t', t=time // self.temporal_compress)

        return emb


class GazeDecoder(nn.Module):
    def __init__(self, *, 
                 out_channels: int = 2, 
                 num_frames: int = 80,
                 dim_tokens: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 sincos_pos_emb: bool = True, 
                 temporal_compress: Optional[int] = None,
                 learnable_pos_emb: bool = False,
                 post_mlp: bool = True,
                 out_conv: bool = False,
                 **ignore_kwargs):
        super().__init__()

        self.out_channels = out_channels
        self.dim_tokens = dim_tokens
        self.temporal_compress = 4 if temporal_compress is None else temporal_compress
        self.position_embeddings = build_1d_sincos_posemb(num_frames // self.temporal_compress, embed_dim=dim_tokens)
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

        self.out_proj = nn.Linear(dim_tokens, self.out_channels * self.temporal_compress)
        
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
        if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv2d, nn.Conv1d)):
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
        B, D, NT = x.shape
        x = x + self.position_embeddings.permute(0, 2, 1)
        x = rearrange(x, 'b d t -> b t d')

        # Transformer forward pass
        x = self.blocks(x)

        # Project each token to (C * P_H * P_W)
        if hasattr(self, 'post_mlp'):
            x = x + self.post_mlp(self.norm_mlp(x))

        x = self.out_proj(x)
        x = rearrange(
            x, 'b nt (c pt) -> b (nt pt) c',
            nt=NT, pt=self.temporal_compress, c=self.out_channels
        )
        return x

