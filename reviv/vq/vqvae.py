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
from typing import List, Tuple, Dict, Optional, Union, Any
from contextlib import nullcontext
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers import StableDiffusionPipeline
from huggingface_hub import PyTorchModelHubMixin

from reviv.vq.quantizers import VectorQuantizerLucid, Memcodes
import reviv.vq.models.vit_models as vit_models 
import reviv.vq.models.vivit_models as vivit_models
import reviv.vq.models.unet.unet as unet
import reviv.vq.models.uvit as uvit
import reviv.vq.models.controlnet as controlnet
from reviv.vq.models.mlp_models import build_mlp
from reviv.vq.models.hand_transformer import HandEncoder, HandDecoder
from reviv.vq.models.body_transformer import BodyEncoder, BodyDecoder
from reviv.vq.models.cam_transformer import CamEncoder, CamDecoder
from reviv.vq.models.gaze_transformer import GazeEncoder, GazeDecoder
from reviv.vq.scheduling import DDPMScheduler, DDIMScheduler, PNDMScheduler, PipelineCond

from reviv.utils import denormalize


# If freeze_enc is True, the following modules will be frozen
FREEZE_MODULES = ['encoder', 'quant_proj', 'quantize', 'cls_emb']

# Root directory for pretrained tokenizer weights. Override with REVIV_DATA_ROOT env var.
DATA_ROOT = os.environ.get("REVIV_DATA_ROOT", "./example_data")


class VQ(nn.Module, PyTorchModelHubMixin):
    """Base class for VQVAE and DiVAE models. Implements the encoder and quantizer, and can be used as such without a decoder
    after training.

    Args:
        image_size: Input and target image size.
        image_size_enc: Input image size for the encoder. Defaults to image_size. Change this when loading weights 
          from a tokenizer trained on a different image size.
        n_channels: Number of input channels.
        n_labels: Number of classes for semantic segmentation.
        enc_type: String identifier specifying the encoder architecture. See vq/vit_models.py and vq/mlp_models.py 
            for available architectures.
        patch_proj: Whether or not to use a ViT-style patch-wise linear projection in the encoder.
        post_mlp: Whether or not to add a small point-wise MLP before the quantizer.
        patch_size: Patch size for the encoder.
        quant_type: String identifier specifying the quantizer implementation. Can be 'lucid', or 'memcodes'.
        codebook_size: Number of codebook entries.
        num_codebooks: Number of "parallel" codebooks to use. Only relevant for 'lucid' and 'memcodes' quantizers.
          When using this, the tokens will be of shape B N_C H_Q W_Q, where N_C is the number of codebooks.
        latent_dim: Dimensionality of the latent code. Can be small when using norm_codes=True, 
          see ViT-VQGAN (https://arxiv.org/abs/2110.04627) paper for details.
        norm_codes: Whether or not to normalize the codebook entries to the unit sphere.
          See ViT-VQGAN (https://arxiv.org/abs/2110.04627) paper for details.
        norm_latents: Whether or not to normalize the latent codes to the unit sphere for computing commitment loss.
        sync_codebook: Enable this when training on multiple GPUs, and disable for single GPUs, e.g. at inference.
        ema_decay: Decay rate for the exponential moving average of the codebook entries.
        threshold_ema_dead_code: Threshold for replacing stale codes that are used less than the 
          indicated exponential moving average of the codebook entries.
        code_replacement_policy: Policy for replacing stale codes. Can be 'batch_random' or 'linde_buzo_gray'.
        commitment_weight: Weight for the quantizer commitment loss.
        kmeans_init: Whether or not to initialize the codebook entries with k-means clustering.
        ckpt_path: Path to a checkpoint to load the model weights from.
        ignore_keys: List of keys to ignore when loading the state_dict from the above checkpoint.
        freeze_enc: Whether or not to freeze the encoder weights. See FREEZE_MODULES for the list of modules.
        undo_std: Whether or not to undo any ImageNet standardization and transform the images to [-1,1] 
          before feeding the input to the encoder.
        config: Dictionary containing the model configuration. Only used when loading
            from Huggingface Hub. Ignore otherwise.
    """
    def __init__(self,
                 image_size: int = 224,
                 n_frames: int = 32,
                 tubelet_size = [2, 16, 16],
                 image_size_enc: Optional[int] = None,
                 n_channels: str = 3,
                 n_labels: Optional[int] = None,
                 enc_type: str = 'vivit_b_enc',
                 patch_proj: bool = True,
                 post_mlp: bool = False,
                 patch_size: int = 16,
                 quant_type: str = 'lucid',
                 codebook_size: Union[int, str] = 16384,
                 num_codebooks: int = 1,
                 latent_dim: int = 32,
                 norm_codes: bool = True,
                 norm_latents: bool = False,
                 sync_codebook: bool = True,
                 ema_decay: float = 0.99,
                 threshold_ema_dead_code: float = 0.25,
                 code_replacement_policy: str = 'batch_random',
                 commitment_weight: float = 1.0,
                 kmeans_init: bool = False,
                 ckpt_path: Optional[str] = None,
                 init_encoder: bool=True,
                 body_joint_nums: int = 22,
                 ignore_keys: List[str] = [
                    'decoder', 'loss', 
                    'post_quant_conv', 'post_quant_proj', 
                    'encoder.pos_emb',
                 ],
                 freeze_enc: bool = False,
                 undo_std: bool = False,
                 config: Optional[Dict[str, Any]] = None,
                 temporal_compress: Optional[int] = None,
                 **kwargs):
        if config is not None:
            config = copy.deepcopy(config)
            self.__init__(**config)
            return
        
        super().__init__()

        self.image_size = image_size
        self.n_channels = n_channels
        self.n_frames = n_frames
        self.tubelet_size = tubelet_size
        self.n_labels = n_labels
        self.enc_type = enc_type
        self.patch_proj = patch_proj
        self.post_mlp = post_mlp
        self.patch_size = patch_size
        self.quant_type = quant_type
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.latent_dim = latent_dim
        self.norm_codes = norm_codes
        self.norm_latents = norm_latents
        self.sync_codebook = sync_codebook
        self.ema_decay = ema_decay
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.code_replacement_policy = code_replacement_policy
        self.commitment_weight = commitment_weight
        self.kmeans_init = kmeans_init
        self.ckpt_path = ckpt_path
        self.body_joint_nums = body_joint_nums
        self.ignore_keys = ignore_keys
        self.freeze_enc = freeze_enc
        self.undo_std = undo_std
        self.temporal_compress = temporal_compress

        # For semantic segmentation
        if n_labels is not None:
            # Q: why embedding_dim this small? 3?
            self.cls_emb = nn.Embedding(num_embeddings=n_labels, embedding_dim=n_channels)
            self.colorize = torch.randn(3, n_labels, 1, 1)
        else:
            self.cls_emb = None

        # Init encoder
        image_size_enc = image_size_enc or image_size
        if 'vivit' in enc_type:
            self.encoder = getattr(vivit_models, enc_type)(
                in_channels=n_channels, num_frames=n_frames, tubelet_size=tubelet_size,
                resolution=image_size_enc, patch_proj=patch_proj, post_mlp=post_mlp,
            )
            self.enc_dim = self.encoder.dim_tokens
        elif 'vit' in enc_type:
            self.encoder = getattr(vit_models, enc_type)(
                in_channels=n_channels, patch_size=patch_size, 
                resolution=image_size_enc, patch_proj=patch_proj, post_mlp=post_mlp
            )
            self.enc_dim = self.encoder.dim_tokens
        elif 'MLP' in enc_type:
            self.encoder = build_mlp(model_id=enc_type, dim_in=n_channels, dim_out=None)
            self.enc_dim = self.encoder.dim_out
        elif 'HandTransformer' in enc_type:
            self.encoder = HandEncoder(num_frames=self.n_frames, tubelet_size=self.tubelet_size)
            self.enc_dim = self.encoder.dim_tokens
        elif 'CamTransformer' in enc_type:
            self.encoder = CamEncoder(num_frames=self.n_frames, temporal_compress=self.temporal_compress)
            self.enc_dim = self.encoder.dim_tokens
        elif 'GazeTransformer' in enc_type:
            self.encoder = GazeEncoder(num_frames=self.n_frames, temporal_compress=self.temporal_compress)
            self.enc_dim = self.encoder.dim_tokens
        elif 'BodyTransformer' in enc_type:
            self.encoder = BodyEncoder(
                num_frames=self.n_frames,
                tubelet_size=self.tubelet_size,
                joint_nums=self.body_joint_nums,
            )
            self.enc_dim = self.encoder.dim_tokens
        else:
            raise NotImplementedError(f'{enc_type} not implemented.')
        
        # Encoder -> quantizer projection
        if 'vivit' in enc_type:
            self.quant_proj = torch.nn.Conv3d(self.enc_dim, self.latent_dim, 1)
        elif 'GAT' in enc_type or 'HandTransformer' in enc_type or 'BodyTransformer' in enc_type:
            self.quant_proj = torch.nn.Conv2d(self.enc_dim, self.latent_dim, 1)
        elif 'CamTransformer' in enc_type or 'GazeTransformer' in enc_type:
            self.quant_proj = torch.nn.Conv1d(self.enc_dim, self.latent_dim, 1)
        else:
            raise NotImplementedError(f'{enc_type} not implemented.')

        # Init quantizer
        if quant_type == 'lucid':
            self.quantize = VectorQuantizerLucid(
                dim=latent_dim,
                codebook_size=codebook_size,
                codebook_dim=latent_dim,
                heads=num_codebooks,
                use_cosine_sim = norm_codes,
                threshold_ema_dead_code = threshold_ema_dead_code,
                code_replacement_policy=code_replacement_policy,
                sync_codebook = sync_codebook,
                decay = ema_decay,
                commitment_weight=self.commitment_weight,
                norm_latents = norm_latents,
                kmeans_init=kmeans_init,
            )
        elif quant_type == 'memcodes':
            pass  # need to adapt this image quantizer to video quantizer

            self.quantize = Memcodes(
                dim=latent_dim, codebook_size=codebook_size,
                heads=num_codebooks, temperature=1.,
            )
        else:
            raise ValueError(f'{quant_type} not a valid quant_type.')

        # Load checkpoint
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            pass

        
        # Freeze encoder
        if freeze_enc:
            for module_name, module in self.named_children():
                if module_name not in FREEZE_MODULES:
                    continue
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()

    def train(self, mode: bool = True) -> 'VQ':
        """Override the default train() to set the training mode to all modules 
        except the encoder if freeze_enc is True.

        Args:
            mode: Whether to set the model to training mode (True) or evaluation mode (False).
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module_name, module in self.named_children():
            if self.freeze_enc and module_name in FREEZE_MODULES:
                continue
            module.train(mode)
        return self

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = list()) -> 'VQ':
        """Loads the state_dict from a checkpoint file and initializes the model with it.
        Renames the keys in the state_dict if necessary (e.g. when loading VQ-GAN weights).

        Args:
            path: Path to the checkpoint file.
            ignore_keys: List of keys to ignore when loading the state_dict.

        Returns:
            self
        """
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt['model'] if 'model' in ckpt else ckpt['state_dict']

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    del sd[k]
        msg = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

        return self

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocesses the input image tensor before feeding it to the encoder.
        If self.undo_std, the input is first denormalized from the ImageNet 
        standardization to [-1, 1]. If semantic segmentation is performed, the 
        class indices are embedded.

        Args:
            x: Input image tensor of shape B C H W 
              or B H W in case of semantic segmentation

        Returns:
            Preprocessed input tensor of shape B C H W
        """
        if self.undo_std:
            x = 2.0 * denormalize(x) - 1.0
        if self.cls_emb is not None:
            pass  # check! x should be float right? how can it be processed with cls_emb?

            x = rearrange(self.cls_emb(x), 'b h w c -> b c h w')
        return x

    def to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """When semantic segmentation is performed, this function converts the 
        class embeddings to RGB.

        Args:
            x: Input tensor of shape B C H W

        Returns:
            RGB tensor of shape B C H W
        """
        x = F.conv2d(x, weight=self.colorize)
        x = (x-x.min())/(x.max()-x.min())
        return x

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """Encodes an input video tensor and quantizes the latent code.

        Args:
            x: Input video tensor of shape B C T H W
              or B T H W in case of semantic segmentation
        
        Returns:
            quant: Quantized latent code of shape B D_Q H_Q W_Q
            code_loss: Codebook loss
            tokens: Quantized indices of shape B T_Q H_Q W_Q
        """
    
        x = self.prepare_input(x)
        h = self.encoder(x)


        h = self.quant_proj(h)
        quant, code_loss, tokens = self.quantize(h)
        return quant, code_loss, tokens

    def tokenize(self, x: torch.Tensor) -> torch.LongTensor:
        """Tokenizes an input image tensor.

        Args:
            x: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation

        Returns:
            Quantized indices of shape B H_Q W_Q
        """
        _, _, tokens = self.encode(x)
        return tokens
    
    def autoencode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Autoencodes an input image tensor by encoding it, quantizing the latent code, 
        and decoding it back to an image.

        Args:
            x: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation
        
        Returns:
            Reconstructed image tensor of shape B C H W
        """
        pass

    def decode_quant(self, quant: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decodes quantized latent codes back to an image.

        Args:
            quant: Quantized latent code of shape B D_Q H_Q W_Q

        Returns:
            Decoded image tensor of shape B C H W
        """
        pass

    def tokens_to_embedding(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Look up the codebook entries corresponding the discrete tokens.

        Args:
            tokens: Quantized indices of shape B H_Q W_Q

        Returns:
            Quantized latent code of shape B D_Q H_Q W_Q
        """
        return self.quantize.indices_to_embedding(tokens)

    def decode_tokens(self, tokens: torch.LongTensor, **kwargs) -> torch.Tensor:
        """Decodes discrete tokens back to an image.

        Args:
            tokens: Quantized indices of shape B H_Q W_Q

        Returns:
            Decoded image tensor of shape B C H W
        """
        quant = self.tokens_to_embedding(tokens)
        dec = self.decode_quant(quant, **kwargs)
        return dec
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder and quantizer.

        Args:
            x: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation
        
        Returns:
            quant: Quantized latent code of shape B D_Q H_Q W_Q
            code_loss: Codebook loss
        """
        quant, code_loss, _ = self.encode(x)
        return quant, code_loss


class VQVAE(VQ):
    """VQ-VAE model = simple encoder + decoder with a discrete bottleneck and 
    basic reconstruction loss (optionall with perceptual loss), i.e. no diffusion, 
    nor GAN discriminator.

    Args:
        dec_type: String identifier specifying the decoder architecture. 
          See vq/vit_models.py and vq/mlp_models.py for available architectures.
        out_conv: Whether or not to add final conv layers to the ViT decoder.
        image_size_dec: Image size for the decoder. Defaults to self.image_size. 
          Change this when loading weights from a tokenizer decoder trained on a 
          different image size.
        patch_size_dec: Patch size for the decoder. Defaults to self.patch_size.
        config: Dictionary containing the model configuration. Only used when loading
            from Huggingface Hub. Ignore otherwise.
    """
    def __init__(self, 
                 dec_type: str = 'vivit_b_dec', 
                 out_conv: bool = False,
                 image_size_dec: int = None, 
                 patch_size_dec: int = None,
                 config: Optional[Dict[str, Any]] = None,
                 *args, 
                 **kwargs):
        if config is not None:
            config = copy.deepcopy(config)
            self.__init__(**config)
            return
        # Don't want to load the weights just yet
        self.original_ckpt_path = kwargs.get('ckpt_path', None)
        kwargs['ckpt_path'] = None
        super().__init__(*args, **kwargs)
        self.ckpt_path = self.original_ckpt_path

        # Init decoder
        out_channels = self.n_channels if self.n_labels is None else self.n_labels
        image_size_dec = image_size_dec or self.image_size
        patch_size = patch_size_dec or self.patch_size
        if 'vivit' in dec_type:
            self.decoder = getattr(vivit_models, dec_type)(
                out_channels=out_channels, num_frames=self.n_frames, tubelet_size=self.tubelet_size,
                resolution=image_size_dec, out_conv=out_conv, post_mlp=self.post_mlp,
                patch_proj=self.patch_proj
            )
            self.dec_dim = self.decoder.dim_tokens
        elif 'vit' in dec_type:
            self.decoder = getattr(vit_models, dec_type)(
                out_channels=out_channels, patch_size=patch_size, 
                resolution=image_size_dec, out_conv=out_conv, post_mlp=self.post_mlp,
                patch_proj=self.patch_proj
            )
            self.dec_dim = self.decoder.dim_tokens
        elif 'MLP' in dec_type:
            self.decoder = build_mlp(model_id=dec_type, dim_in=None, dim_out=out_channels)
            self.dec_dim = self.decoder.dim_in
        elif 'HandTransformer' in dec_type:
            self.decoder = HandDecoder(num_frames=self.n_frames, tubelet_size=self.tubelet_size)
            self.dec_dim = self.decoder.dim_tokens
        elif 'BodyTransformer' in dec_type:
            self.decoder = BodyDecoder(
                num_frames=self.n_frames,
                tubelet_size=self.tubelet_size,
                joint_nums=self.body_joint_nums,
            )
            self.dec_dim = self.decoder.dim_tokens
        elif 'CamTransformer' in dec_type:
            self.decoder = CamDecoder(num_frames=self.n_frames, temporal_compress=self.temporal_compress)
            self.dec_dim = self.decoder.dim_tokens
        elif 'GazeTransformer' in dec_type:
            self.decoder = GazeDecoder(num_frames=self.n_frames, temporal_compress=self.temporal_compress)
            self.dec_dim = self.decoder.dim_tokens
        else:
            raise NotImplementedError(f'{dec_type} not implemented.')

        # Quantizer -> decoder projection
        if 'vivit' in dec_type:
            self.post_quant_proj = torch.nn.Conv3d(self.latent_dim, self.dec_dim, 1)
        elif 'GAT' in dec_type or 'HandTransformer' in dec_type or 'BodyTransformer' in dec_type:
            self.post_quant_proj = torch.nn.Conv2d(self.latent_dim, self.dec_dim, 1)
        elif 'CamTransformer' in dec_type or 'GazeTransformer' in dec_type:
            self.post_quant_proj = torch.nn.Conv1d(self.latent_dim, self.dec_dim, 1)
        else:
            raise NotImplementedError(f'{dec_type} not implemented.')
        
        # Load checkpoint
        if self.ckpt_path is not None:
            self.init_from_ckpt(self.ckpt_path, ignore_keys=self.ignore_keys)
            pass

        
        if kwargs.get('init_encoder', False):
            self._init_weights_from_pretrained()
            print('Init weights from ReViV!')

    def _init_encoder_weights_from_vivit(self):
        pass

        from transformers import VivitModel
        pretrained_model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400").state_dict()
        my_model_dict = self.state_dict()

        # init conv3d tubelet proj
        name = 'encoder.embeddings.patch_embeddings.projection.weight'
        my_model_dict[name].copy_(pretrained_model[name])
        name = 'encoder.embeddings.patch_embeddings.projection.bias'
        my_model_dict[name].copy_(pretrained_model[name])

        # init transformer blocks
        for blk_id in range(self.encoder.depth):
            name = 'encoder.layer.%d.layernorm_before.weight' % blk_id
            my_model_dict['blocks.%d.norm1.weight' % blk_id].copy_(pretrained_model[name])
            name = 'encoder.layer.%d.layernorm_before.bias' % blk_id
            my_model_dict['blocks.%d.norm1.bias' % blk_id].copy_(pretrained_model[name])

            name = 'encoder.layer.%d.attention.attention.query.weight' % blk_id
            q_w = pretrained_model[name]
            name = 'encoder.layer.%d.attention.attention.key.weight' % blk_id
            k_w = pretrained_model[name]
            name = 'encoder.layer.%d.attention.attention.value.weight' % blk_id
            v_w = pretrained_model[name]
            my_model_dict['blocks.%d.attn.qkv.weight' % blk_id].copy_(torch.cat([q_w, k_w, v_w], dim=0))

            name = 'encoder.layer.%d.attention.attention.query.bias' % blk_id
            q_b = pretrained_model[name]
            name = 'encoder.layer.%d.attention.attention.key.bias' % blk_id
            k_b = pretrained_model[name]
            name = 'encoder.layer.%d.attention.attention.value.bias' % blk_id
            v_b = pretrained_model[name]
            my_model_dict['blocks.%d.attn.qkv.bias' % blk_id].copy_(torch.cat([q_b, k_b, v_b], dim=0))

            name = 'encoder.layer.%d.attention.output.dense.weight' % blk_id
            my_model_dict['blocks.%d.attn.proj.weight' % blk_id].copy_(pretrained_model[name])
            name = 'encoder.layer.%d.attention.output.dense.bias' % blk_id
            my_model_dict['blocks.%d.attn.proj.bias' % blk_id].copy_(pretrained_model[name])

            name = 'encoder.layer.%d.layernorm_after.weight' % blk_id
            my_model_dict['blocks.%d.norm2.weight' % blk_id].copy_(pretrained_model[name])
            name = 'encoder.layer.%d.layernorm_after.bias' % blk_id
            my_model_dict['blocks.%d.norm2.bias' % blk_id].copy_(pretrained_model[name])

            name = 'encoder.layer.%d.intermediate.dense.weight' % blk_id
            my_model_dict['blocks.%d.mlp.fc1.weight' % blk_id].copy_(pretrained_model[name])
            name = 'encoder.layer.%d.intermediate.dense.bias' % blk_id
            my_model_dict['blocks.%d.mlp.fc1.bias' % blk_id].copy_(pretrained_model[name])

            name = 'encoder.layer.%d.output.dense.weight' % blk_id
            my_model_dict['blocks.%d.mlp.fc2.weight' % blk_id].copy_(pretrained_model[name])
            name = 'encoder.layer.%d.output.dense.bias' % blk_id
            my_model_dict['blocks.%d.mlp.fc2.bias' % blk_id].copy_(pretrained_model[name])
        
        self.load_state_dict(my_model_dict)
        return self
    
    def _init_weights_from_pretrained(self):
        from reviv.vq.vqvae_reviv import DiVAE
        from reviv.utils import load_safetensors
        # avoid network issues...
        ckpt, config = load_safetensors(f'{DATA_ROOT}/rgb_tok_pretrained.safetensors')
        pretrained_model = DiVAE(config=config)
        pretrained_model.load_state_dict(ckpt)
        pretrained_model = pretrained_model.state_dict()
        my_model_dict = self.state_dict()

        # init conv3d tubelet proj from conv2d: central frame init
        name = 'encoder.embeddings.patch_embeddings.projection.weight'
        t = my_model_dict[name].shape[2] // 2
        my_model_dict[name][:, :, t].copy_(pretrained_model['encoder.proj.weight'])
        name = 'encoder.embeddings.patch_embeddings.projection.bias'
        my_model_dict[name].copy_(pretrained_model['encoder.proj.bias'])

        # init transformer blocks
        for blk_id in range(self.encoder.depth):
            my_model_dict['encoder.blocks.%d.norm1.weight' % blk_id].copy_(pretrained_model['encoder.blocks.%d.norm1.weight' % blk_id])
            my_model_dict['encoder.blocks.%d.norm1.bias' % blk_id].copy_(pretrained_model['encoder.blocks.%d.norm1.bias' % blk_id])

            my_model_dict['encoder.blocks.%d.attn.qkv.weight' % blk_id].copy_(pretrained_model['encoder.blocks.%d.attn.qkv.weight' % blk_id])
            my_model_dict['encoder.blocks.%d.attn.qkv.bias' % blk_id].copy_(pretrained_model['encoder.blocks.%d.attn.qkv.bias' % blk_id])

            my_model_dict['encoder.blocks.%d.attn.proj.weight' % blk_id].copy_(pretrained_model['encoder.blocks.%d.attn.proj.weight' % blk_id])
            my_model_dict['encoder.blocks.%d.attn.proj.bias' % blk_id].copy_(pretrained_model['encoder.blocks.%d.attn.proj.bias' % blk_id])

            my_model_dict['encoder.blocks.%d.norm2.weight' % blk_id].copy_(pretrained_model['encoder.blocks.%d.norm2.weight' % blk_id])
            my_model_dict['encoder.blocks.%d.norm2.bias' % blk_id].copy_(pretrained_model['encoder.blocks.%d.norm2.bias' % blk_id])

            my_model_dict['encoder.blocks.%d.mlp.fc1.weight' % blk_id].copy_(pretrained_model['encoder.blocks.%d.mlp.fc1.weight' % blk_id])
            my_model_dict['encoder.blocks.%d.mlp.fc1.bias' % blk_id].copy_(pretrained_model['encoder.blocks.%d.mlp.fc1.bias' % blk_id])

            my_model_dict['encoder.blocks.%d.mlp.fc2.weight' % blk_id].copy_(pretrained_model['encoder.blocks.%d.mlp.fc2.weight' % blk_id])
            my_model_dict['encoder.blocks.%d.mlp.fc2.bias' % blk_id].copy_(pretrained_model['encoder.blocks.%d.mlp.fc2.bias' % blk_id])
        
        my_model_dict['encoder.norm_mlp.weight'].copy_(pretrained_model['encoder.norm_mlp.weight'])
        my_model_dict['encoder.norm_mlp.bias'].copy_(pretrained_model['encoder.norm_mlp.bias'])

        my_model_dict['encoder.post_mlp.fc1.weight'].copy_(pretrained_model['encoder.post_mlp.fc1.weight'])
        my_model_dict['encoder.post_mlp.fc1.bias'].copy_(pretrained_model['encoder.post_mlp.fc1.bias'])
        my_model_dict['encoder.post_mlp.fc2.weight'].copy_(pretrained_model['encoder.post_mlp.fc2.weight'])
        my_model_dict['encoder.post_mlp.fc2.bias'].copy_(pretrained_model['encoder.post_mlp.fc2.bias'])

        # init weights for quantizer projection
        my_model_dict['quant_proj.weight'][:, :, 0].copy_(pretrained_model['quant_proj.weight']) # init conv3d from conv2d
        my_model_dict['quant_proj.bias'].copy_(pretrained_model['quant_proj.bias'])
        
        self.load_state_dict(my_model_dict)
        return self

    def decode_quant(self, quant: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decodes quantized latent codes back to an image.

        Args:
            quant: Quantized latent code of shape B D_Q H_Q W_Q

        Returns:
            Decoded image tensor of shape B C H W
        """
        quant = self.post_quant_proj(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder, quantizer, and decoder.

        Args:
            x: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation
        
        Returns:
            dec: Decoded image tensor of shape B C H W
            code_loss: Codebook loss
        """
        with torch.no_grad() if self.freeze_enc else nullcontext():
            quant, code_loss, _ = self.encode(x)
        dec = self.decode_quant(quant)
        return dec, code_loss

    def autoencode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Autoencodes an input image tensor by encoding it, quantizing the 
        latent code, and decoding it back to an image.

        Args:
            x: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation
        
        Returns:
            Reconstructed image tensor of shape B C H W
        """
        dec, _ = self.forward(x)
        return dec


class DiVAE(VQ):
    """DiVAE ("Diffusion VQ-VAE") model = simple encoder + diffusion decoder with 
    a discrete bottleneck, inspired by https://arxiv.org/abs/2206.00386. 
    
    Args:
        dec_type: String identifier specifying the decoder architecture.
          See vq/models/unet/unet.py and vq/models/uvit.py for available architectures.
        num_train_timesteps: Number of diffusion timesteps to use for training.
        cls_free_guidance_dropout: Dropout probability for classifier-free guidance.
        masked_cfg: Whether or not to randomly mask out conditioning tokens.
          cls_free_guidance_dropout must be > 0.0 for this to have any effect, and
          decides how often masking is performed. E.g. with 0.5, half of the time
          the conditioning tokens will be randomly masked, and half the time they
          will be kept as is.
        masked_cfg_low: Lower bound of number of tokens to mask out.
        masked_cfg_high: Upper bound of number of tokens to mask out (inclusive).
          Defaults to total number of tokens (T_Q * H_Q * W_Q) if it is set to None.
        scheduler: String identifier specifying the diffusion scheduler to use.
            Can be 'ddpm' or 'ddim'.
        beta_schedule: String identifier specifying the beta schedule to use for 
          the diffusion process. Can be 'linear', 'squaredcos_cap_v2' (cosine), 
          'shifted_cosine:{shift_amount}'; see vq/scheduling for details.
        prediction_type: String identifier specifying the type of prediction to use.
          Can be 'sample', 'epsilon', or 'v_prediction'; see vq/scheduling for details.
        clip_sample: Whether or not to clip the samples to [-1, 1], at inference only.
        thresholding: Whether or not to use dynamic thresholding  (introduced by Imagen, 
          https://arxiv.org/abs/2205.11487) for the diffusion process, at inference only.
        conditioning: String identifier specifying the way to condition the diffusion 
          decoder. Can be 'concat' or 'xattn'. See models for details (only relevant to UViT).
        dec_transformer_dropout: Dropout rate for the transformer layers in the 
          diffusion decoder (only relevant to UViT models).
        zero_terminal_snr: Whether or not to enforce zero terminal SNR, i.e. the SNR
          at the last timestep is set to zero. This is useful for preventing the model 
          from "cheating" by using information in the last timestep to reconstruct the image.
          See https://arxiv.org/abs/2305.08891.
        image_size_dec: Image size for the decoder. Defaults to image_size. 
          Change this when loading weights from a tokenizer decoder trained on a 
          different image size.
        config: Dictionary containing the model configuration. Only used when loading
            from Huggingface Hub. Ignore otherwise.
    """
    def __init__(self, 
                 dec_type: str = 'unet_patched',
                 num_train_timesteps: int = 1000, 
                 cls_free_guidance_dropout: float = 0.0,
                 masked_cfg: bool = False, 
                 masked_cfg_low: int = 0,
                 masked_cfg_high: Optional[int] = None,
                 scheduler: str = 'ddpm',
                 beta_schedule: str = 'squaredcos_cap_v2',
                 prediction_type: str = 'v_prediction',
                 clip_sample: bool = False, 
                 thresholding: bool = True, 
                 conditioning: str = 'concat',
                 dec_transformer_dropout: float = 0.2,
                 zero_terminal_snr: bool = True,
                 image_size_dec: Optional[int] = None,
                 config: Optional[Dict[str, Any]] = None,
                 *args, **kwargs):
        if config is not None:
            config = copy.deepcopy(config)
            self.__init__(**config)
            return
        # Don't want to load the weights just yet
        self.original_ckpt_path = kwargs.get('ckpt_path', None)
        kwargs['ckpt_path'] = None
        super().__init__(*args, **kwargs)
        self.ckpt_path = self.original_ckpt_path
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.thresholding = thresholding
        self.zero_terminal_snr = zero_terminal_snr

        if cls_free_guidance_dropout > 0.0:
            self.cfg_dist = torch.distributions.Bernoulli(probs=cls_free_guidance_dropout)
        else:
            self.cfg_dist = None
        self.masked_cfg = masked_cfg
        self.masked_cfg_low = masked_cfg_low
        self.masked_cfg_high = masked_cfg_high

        # Init diffusion decoder
        image_size_dec = image_size_dec or self.image_size
        if 'unet_' in dec_type:
            self.decoder = getattr(unet, dec_type)(
                in_channels=self.n_channels, 
                out_channels=self.n_channels, 
                cond_channels=self.latent_dim, 
                image_size=image_size_dec,
            )
        elif 'uvit_' in dec_type:
            self.decoder = getattr(uvit, dec_type)(
                sample_size=image_size_dec,
                in_channels=self.n_channels,
                out_channels=self.n_channels,
                cond_dim=self.latent_dim,
                cond_type=conditioning,
                mid_drop_rate=dec_transformer_dropout,
            )
        else:
            raise NotImplementedError(f'dec_type {dec_type} not implemented.')
        
        # Init training diffusion scheduler / default pipeline for generation
        scheduler_cls = DDPMScheduler if scheduler == 'ddpm' else DDIMScheduler
        self.noise_scheduler = scheduler_cls(
            num_train_timesteps=num_train_timesteps, 
            thresholding=thresholding,
            clip_sample=clip_sample,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            zero_terminal_snr=zero_terminal_snr,
        )
        self.pipeline = PipelineCond(model=self.decoder, scheduler=self.noise_scheduler)

        # Load checkpoint
        if self.ckpt_path is not None:
            self.init_from_ckpt(self.ckpt_path, ignore_keys=self.ignore_keys)
        
    
    def _init_weights_from_pretrained(self):
        from reviv.vq.vqvae_reviv import DiVAE
        pretrained_model = DiVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_rgb_16k_224-448').state_dict()
        my_model_dict = self.state_dict()

        # init weights for diffusion decorder
        keys = list(pretrained_model.keys())
        for k in keys:
            for ik in FREEZE_MODULES:
                if k.startswith(ik):
                    del pretrained_model[k]
        pass

        # convert conv2d weight to conv3d weight, and decoder.out[2] bias
        # without loading weights is already good enough according to the loss

        msg = self.load_state_dict(pretrained_model, strict=False)
        
        return self

    def sample_mask(self, quant: torch.Tensor, low: int = 0, high: Optional[int] = None) -> torch.BoolTensor:
        """Returns a mask of shape B H_Q W_Q, where True = masked-out, False = keep.

        Args:
            quant: Dequantized latent tensor of shape B D_Q H_Q W_Q
            low: Lower bound of number of tokens to mask out
            high: Upper bound of number of tokens to mask out (inclusive). 
              Defaults to total number of tokens (H_Q * W_Q) if it is set to None.

        Returns:
            Boolean mask of shape B H_Q W_Q
        """
        B, _, T_Q, H_Q, W_Q = quant.shape
        num_tokens = T_Q * H_Q * W_Q
        high = high if high is not None else num_tokens
        
        zero_idxs = torch.randint(low=low, high=high+1, size=(B,), device=quant.device)
        noise = torch.rand(B, num_tokens, device=quant.device)
        ids_arange_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        mask = torch.where(ids_arange_shuffle < zero_idxs.unsqueeze(1), 0, 1)
        mask = rearrange(mask, 'b (t h w) -> b t h w', t=T_Q, h=H_Q, w=W_Q).bool()
        
        return mask

    def _get_pipeline(self, scheduler: Optional[SchedulerMixin] = None) -> PipelineCond:
        """Creates a conditional diffusion pipeline with the given scheduler.

        Args:
            scheduler: Scheduler to use for the diffusion pipeline.
              If None, the default scheduler will be used.

        Returns:
            Conditional diffusion pipeline.
        """
        return PipelineCond(model=self.decoder, scheduler=scheduler) if scheduler is not None else self.pipeline

    def decode_quant(self, 
                     quant: torch.Tensor, 
                     timesteps: Optional[int] = None, 
                     scheduler: Optional[SchedulerMixin] = None, 
                     generator: Optional[torch.Generator] = None, 
                     image_size: Optional[Union[Tuple[int, int], int]] = None, 
                     num_frames: Optional[int] = None,
                     verbose: bool = False, 
                     scheduler_timesteps_mode: str = 'trailing', 
                     orig_res: Optional[Union[torch.LongTensor, Tuple[int, int]]] = None) -> torch.Tensor:
        """Decodes quantized latent codes back to an image.

        Args:
            quant: Quantized latent code of shape B D_Q H_Q W_Q
            timesteps: Number of diffusion timesteps to use. Defaults to self.num_train_timesteps.
            scheduler: Scheduler to use for the diffusion pipeline. Defaults to the training scheduler.
            generator: Random number generator to use for sampling. By default generations are stochastic.
            image_size: Image size to use for the diffusion pipeline. Defaults to decoder image size.
            verbose: Whether or not to print progress bar.
            scheduler_timesteps_mode: The mode to use for DDIMScheduler. One of `trailing`, `linspace`, 
              `leading`. See https://arxiv.org/abs/2305.08891 for more details.
            orig_res: The original resolution of the image to condition the diffusion on. Ignored if None.
              See SDXL https://arxiv.org/abs/2307.01952 for more details.

        Returns:
            Decoded image tensor of shape B C H W
        """
        pipeline = self._get_pipeline(scheduler)
        dec = pipeline(
            quant, timesteps=timesteps, generator=generator, image_size=image_size, num_frames=num_frames,
            verbose=verbose, scheduler_timesteps_mode=scheduler_timesteps_mode, orig_res=orig_res
        )
        return dec
    
    def decode_tokens(self, tokens: torch.LongTensor, **kwargs) -> torch.Tensor:
        """See `decode_quant` for details on the optional args."""
        return super().decode_tokens(tokens, **kwargs)

    def autoencode(self, 
                   input_clean: torch.Tensor, 
                   timesteps: Optional[int] = None, 
                   scheduler: Optional[SchedulerMixin] = None, 
                   generator: Optional[torch.Generator] = None, 
                   verbose: bool = True, 
                   scheduler_timesteps_mode: str = 'trailing', 
                   orig_res: Optional[Union[torch.LongTensor, Tuple[int, int]]] = None, 
                   **kwargs) -> torch.Tensor:
        """Autoencodes an input image tensor by encoding it, quantizing the latent code, 
            and decoding it back to an image.

        Args:
            input_clean: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation
            timesteps: Number of diffusion timesteps to use. Defaults to self.num_train_timesteps.
            scheduler: Scheduler to use for the diffusion pipeline. Defaults to the training scheduler.
            generator: Random number generator to use for sampling. By default generations are stochastic.
            verbose: Whether or not to print progress bar.
            scheduler_timesteps_mode: The mode to use for DDIMScheduler. One of `trailing`, `linspace`, 
              `leading`. See https://arxiv.org/abs/2305.08891 for more details.
            orig_res: The original resolution of the image to condition the diffusion on. Ignored if None.
              See SDXL https://arxiv.org/abs/2307.01952 for more details.
        
        Returns:
            Reconstructed image tensor of shape B C H W
        """
        pipeline = self._get_pipeline(scheduler)
        quant, _, _ = self.encode(input_clean)
        image_size = input_clean.shape[-1]
        num_frames = input_clean.shape[-3]
        dec = pipeline(
            quant, timesteps=timesteps, generator=generator, image_size=image_size, num_frames=num_frames,
            verbose=verbose, scheduler_timesteps_mode=scheduler_timesteps_mode, orig_res=orig_res
        )
        return dec

    def forward(self, 
                input_clean: torch.Tensor, 
                input_noised: torch.Tensor, 
                timesteps: Union[torch.Tensor, float, int], 
                cond_mask: Optional[torch.Tensor] = None, 
                orig_res: Optional[Union[torch.LongTensor, Tuple[int, int]]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder, quantizer, and decoder.

        Args:
            input_clean: Clean input image tensor of shape B C H W
              or B H W in case of semantic segmentation. Used for encoding.
            input_noised: Noised input image tensor of shape B C H W. Used as 
              input to the diffusion decoder.
            timesteps: Timesteps for conditioning the diffusion decoder on.
            cond_mask: Optional mask for the diffusion conditioning. 
              True = masked-out, False = keep.
            orig_res: The original resolution of the image to condition the diffusion on. Ignored if None.
              See SDXL https://arxiv.org/abs/2307.01952 for more details.
        
        Returns:
            dec: Decoded image tensor of shape B C H W
            code_loss: Codebook loss
        """
        with torch.no_grad() if self.freeze_enc else nullcontext():
            quant, code_loss, _ = self.encode(input_clean)

        if cond_mask is None and self.cfg_dist is not None and self.training:
            # Create a random mask for each batch element. True = masked-out, False = keep
            B, _, T_Q, H_Q, W_Q = quant.shape
            cond_mask = self.cfg_dist.sample((B,)).to(quant.device, dtype=torch.bool)
            cond_mask = repeat(cond_mask, 'b -> b t h w', t=T_Q, h=H_Q, w=W_Q)
            if self.masked_cfg:
                mask = self.sample_mask(quant, low=self.masked_cfg_low, high=self.masked_cfg_high)
                cond_mask = (mask * cond_mask)
        
        dec = self.decoder(input_noised, timesteps, quant, cond_mask=cond_mask, orig_res=orig_res)
        return dec, code_loss


class VQControlNet(VQ):
    """VQControlNet model = simple pertrained encoder + a ControlNet decoder conditioned on tokens.
    
    Args:
        sd_path: Path to the Stable Diffusion weights for training the ControlNet.
        image_size_sd: Stable diffusion input image size. Defaults to image_size.
            Change this to the image size that Stable Diffusion is trained on.
        pretrained_cn: Whether to use pretrained Stable Diffusion weights for the control model.
        cls_free_guidance_dropout: Dropout probability for classifier-free guidance.
        masked_cfg: Whether or not to randomly mask out conditioning tokens.
          cls_free_guidance_dropout must be > 0.0 for this to have any effect, and
          decides how often masking is performed. E.g. with 0.5, half of the time
          the conditioning tokens will be randomly masked, and half the time they
          will be kept as is.
        masked_cfg_low: Lower bound of number of tokens to mask out.
        masked_cfg_high: Upper bound of number of tokens to mask out (inclusive).
          Defaults to total number of tokens (H_Q * W_Q) if it is set to None.
        enable_xformer: Enables xFormers.
        adapter: Path to the adapter model weights. The adapter model is initialy trained to map
            the tokens to a VAE latent-like representation. Then the output of the adapter model
            is passed as the condition to train the ControlNet. By default there is no adapter usage.
        config: Dictionary containing the model configuration. Only used when loading
            from Huggingface Hub. Ignore otherwise.
    """
    def __init__(self,  
                 sd_path: str = "runwayml/stable-diffusion-v1-5",
                 image_size_sd: Optional[int] = None,
                 pretrained_cn: bool = False,
                 cls_free_guidance_dropout: float = 0.0,
                 masked_cfg: bool = False, 
                 masked_cfg_low: int = 0,
                 masked_cfg_high: Optional[int] = None,
                 enable_xformer: bool = False,
                 adapter: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 *args, **kwargs):
        if config is not None:
            config = copy.deepcopy(config)
            self.__init__(**config)
            return
        # Don't want to load the weights just yet
        self.original_ckpt_path = kwargs.get('ckpt_path', None)
        kwargs['ckpt_path'] = None
        super().__init__(*args, **kwargs)
        self.ckpt_path = self.original_ckpt_path

        if cls_free_guidance_dropout > 0.0:
            self.cfg_dist = torch.distributions.Bernoulli(probs=cls_free_guidance_dropout)
        else:
            self.cfg_dist = None
        self.masked_cfg = masked_cfg
        self.masked_cfg_low = masked_cfg_low
        self.masked_cfg_high = masked_cfg_high
        self.image_size_sd = self.image_size if image_size_sd is None else image_size_sd

        sd_pipeline = StableDiffusionPipeline.from_pretrained(sd_path)
        try:
            import xformers
            XFORMERS_AVAILABLE = True
        except ImportError:
            print("xFormers not available")
            XFORMERS_AVAILABLE = False
        enable_xformer = enable_xformer and XFORMERS_AVAILABLE
        if enable_xformer:
            print('Enabling xFormer for Stable Diffusion')
            sd_pipeline.enable_xformers_memory_efficient_attention()

        self.decoder = getattr(controlnet, 'controlnet')(
            in_channels=4, 
            cond_channels=self.latent_dim,
            sd_pipeline=sd_pipeline,
            image_size=self.image_size_sd,
            pretrained_cn=pretrained_cn,
            enable_xformer=enable_xformer,
            adapter=adapter,
        )
        
        # Use the defualt controlnet pipeline both for training and generation
        self.noise_scheduler = PNDMScheduler(**sd_pipeline.scheduler.config)
        self.vae = sd_pipeline.vae
        self._freeze_vae()
        
        self.pipeline = PipelineCond(model=self.decoder, scheduler=self.noise_scheduler)

        # Load checkpoint
        if self.ckpt_path is not None:
            self.init_from_ckpt(self.ckpt_path, ignore_keys=self.ignore_keys)
            pass


    def sample_mask(self, quant: torch.Tensor, low: int = 0, high: Optional[int] = None) -> torch.BoolTensor:
        """Returns a mask of shape B H_Q W_Q, where True = masked-out, False = keep.

        Args:
            quant: Dequantized latent tensor of shape B D_Q H_Q W_Q
            low: Lower bound of number of tokens to mask out
            high: Upper bound of number of tokens to mask out (inclusive). 
              Defaults to total number of tokens (H_Q * W_Q) if it is set to None.

        Returns:
            Boolean mask of shape B H_Q W_Q
        """
        B, _, H_Q, W_Q = quant.shape
        num_tokens = H_Q * W_Q
        high = high if high is not None else num_tokens
        
        zero_idxs = torch.randint(low=low, high=high+1, size=(B,), device=quant.device)
        noise = torch.rand(B, num_tokens, device=quant.device)
        ids_arange_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        mask = torch.where(ids_arange_shuffle < zero_idxs.unsqueeze(1), 0, 1)
        mask = rearrange(mask, 'b (h w) -> b h w', h=H_Q, w=W_Q).bool()
        
        return mask

    def decode_quant(self, 
                     quant: torch.Tensor, 
                     timesteps: Optional[int] = None, 
                     generator: Optional[torch.Generator] = None, 
                     image_size: Optional[Union[Tuple[int, int], int]] = None, 
                     verbose: bool = False, 
                     vae_decode: bool = False,
                     scheduler_timesteps_mode: str = 'leading', 
                     prompt: Optional[Union[List[str], str]]= None,
                     orig_res: Optional[Union[torch.LongTensor, Tuple[int, int]]] = None,
                     guidance_scale: int = 0.0, 
                     cond_scale: int = 1.0) -> torch.Tensor:
        """Decodes quantized latent codes back to an image.

        Args:
            quant: Quantized latent code of shape B D_Q H_Q W_Q
            timesteps: Number of diffusion timesteps to use. Defaults to self.num_train_timesteps.
            generator: Random number generator to use for sampling. By default generations are stochastic.
            image_size: Image size to use for the diffusion pipeline. Defaults to decoder image size.
            verbose: Whether or not to print progress bar.
            vae_decode: If set to True decodes the latent output of stable diffusion
            scheduler_timesteps_mode: The mode to use for DDIMScheduler. One of `trailing`, `linspace`, 
              `leading`. See https://arxiv.org/abs/2305.08891 for more details.
            prompt: the input prompts for controlnet.
            orig_res: The original resolution of the image to condition the diffusion on. Ignored if None.
              See SDXL https://arxiv.org/abs/2307.01952 for more details.
            guidance_scale: Classifier free guidance scale.
            cond_scale: Scale that is multiplied by the output of control model before being added 
                to stable diffusion layers in controlnet.

        Returns:
            Decoded tensor of shape B C H W
        """
        dec = self.pipeline(
            quant, timesteps=timesteps, generator=generator, image_size=image_size, 
            verbose=verbose, scheduler_timesteps_mode=scheduler_timesteps_mode, prompt=prompt,
            guidance_scale=guidance_scale, cond_scale=cond_scale,
        )

        if vae_decode:
            return self.vae_decode(dec)

        return dec

    def decode_tokens(self, tokens: torch.LongTensor, **kwargs) -> torch.Tensor:
        """See `decode_quant` for details on the optional args."""
        return super().decode_tokens(tokens, **kwargs)

    @torch.no_grad()
    def vae_encode(self, x: torch.Tensor):
        """Encodes the input image into vae latent representaiton.
        
        Args:
            x: Input images

        Returns:
           Encoded latent tensor of shape B C H W 
        """
        z = self.vae.encode(x).latent_dist.sample()
        z = z * self.vae.config.scaling_factor
        return z
    
    @torch.no_grad()
    def vae_decode(self, x: torch.Tensor, clip: bool = True) -> torch.Tensor:
        """Decodes the vae latent representation into vae latent representaiton.
        
        Args:
            x: VAE latent representation
            clip: If set True clips the decoded image between -1 and 1.

        Returns:
           Decoded image of shape B C H W 
        """
        x = self.vae.decode(x / self.vae.config.scaling_factor).sample
        if clip:
            x = torch.clip(x, min=-1, max=1)
        return x
    
    def autoencode(self, 
                   input_clean: torch.Tensor, 
                   timesteps: Optional[int] = None, 
                   generator: Optional[torch.Generator] = None, 
                   image_size: Optional[Union[Tuple[int, int], int]] = None, 
                   verbose: bool = False, 
                   vae_decode: bool = False,
                   scheduler_timesteps_mode: str = 'leading', 
                   prompt: Optional[Union[List[str], str]]= None,
                   orig_res: Optional[Union[torch.LongTensor, Tuple[int, int]]] = None,
                   guidance_scale: int = 0.0, 
                   cond_scale: int = 1.0) -> torch.Tensor:
        """Autoencodes an input image tensor by encoding it, quantizing the latent code, 
            and decoding it back to an image.

        Args:
            input_clean: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation
            timesteps: Number of diffusion timesteps to use. Defaults to self.num_train_timesteps.
            scheduler: Scheduler to use for the diffusion pipeline. Defaults to the training scheduler.
            generator: Random number generator to use for sampling. By default generations are stochastic.
            image_size: Image size to use for the diffusion pipeline. Defaults to decoder image size.
            verbose: Whether or not to print progress bar.
            vae_decode: If set to True, decodes the latent output of stable diffusion
            scheduler_timesteps_mode: The mode to use for DDIMScheduler. One of `trailing`, `linspace`, 
              `leading`. See https://arxiv.org/abs/2305.08891 for more details.
            prompt: the input prompts for controlnet.
            orig_res: The original resolution of the image to condition the diffusion on. Ignored if None.
              See SDXL https://arxiv.org/abs/2307.01952 for more details.
            guidance_scale: Classifier free guidance scale.
            cond_scale: Scale that is multiplied by the output of control model before being added 
                to stable diffusion layers in controlnet.
        
        Returns:
            Reconstructed tensor of shape B C H W
        """
        quant, _, _ = self.encode(input_clean)
        dec = self.pipeline(
            quant, timesteps=timesteps, generator=generator,
            verbose=verbose, scheduler_timesteps_mode=scheduler_timesteps_mode, prompt=prompt,
            guidance_scale=guidance_scale, cond_scale=cond_scale,
        )

        if vae_decode:
            return self.vae_decode(dec)

        return dec

    def forward(self, 
                input_clean: torch.Tensor, 
                input_noised: torch.Tensor, 
                timesteps: Union[torch.Tensor, float, int], 
                cond_mask: Optional[torch.Tensor] = None, 
                prompt: Optional[Union[List[str], str]] = None,
                orig_res: Optional[Union[torch.LongTensor, Tuple[int, int]]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder, quantizer, and decoder.

        Args:
            input_clean: Clean input image tensor of shape B C H W
              or B H W in case of semantic segmentation. Used for encoding.
            input_noised: Noised input image tensor of shape B C H W. Used as 
              input to the diffusion decoder.
            timesteps: Timesteps for conditioning the diffusion decoder on.
            cond_mask: Optional mask for the diffusion conditioning. 
              True = masked-out, False = keep.
            prompt: ControlNet input prompt. Defaults to an empty string.
            orig_res: The original resolution of the image to condition the diffusion on. Ignored if None.
              See SDXL https://arxiv.org/abs/2307.01952 for more details.
        
        Returns:
            dec: Decoded image tensor of shape B C H W
            code_loss: Codebook loss
        """
        with torch.no_grad() if self.freeze_enc else nullcontext():
            quant, code_loss, _ = self.encode(input_clean)

        if cond_mask is None and self.cfg_dist is not None and self.training:
            # Create a random mask for each batch element. True = masked-out, False = keep
            B, _, H_Q, W_Q = quant.shape
            cond_mask = self.cfg_dist.sample((B,)).to(quant.device, dtype=torch.bool)
            cond_mask = repeat(cond_mask, 'b -> b h w', h=H_Q, w=W_Q)
            if self.masked_cfg:
                mask = self.sample_mask(quant, low=self.masked_cfg_low, high=self.masked_cfg_high)
                cond_mask = (mask * cond_mask)
        
        dec = self.decoder(input_noised, timesteps, quant, cond_mask=cond_mask, orig_res=orig_res, prompt=prompt)
        return dec, code_loss

    def _freeze_vae(self):
        """Freezes VAE"""
        for param in self.vae.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    """
    model = VQVAE(
        image_size=224,
        n_channels=3,
        enc_type='vivit_b_enc',
        dec_type='vivit_b_dec',
        post_mlp=True,
        patch_proj=True,
        n_frames=32,
        tubelet_size=[2, 16, 16],
        quant_type='lucid',
        codebook_size=16384,
        num_codebooks=1,
        latent_dim=32,
        norm_codes=True,
        norm_latents=False,
        sync_codebook=False,
        ema_decay=0.99,
        threshold_ema_dead_code=0.35,
        code_replacement_policy='batch_random',
        commitment_weight=1.0,
        kmeans_init=False,
        undo_std=False,
        freeze_enc=False,
        out_conv=False,
    ).cuda()
    x = torch.rand(4, 3, 32, 224, 224).cuda()
    y, loss = model(x)
    print(y.shape)
    print(loss)
    print(str(model))
    """
    """
    model = DiVAE(
        image_size=224,
        n_channels=3,
        enc_type='vivit_b_enc',
        dec_type='unet_patched',
        post_mlp=True,
        patch_proj=True,
        n_frames=32,
        tubelet_size=[2, 16, 16],
        quant_type='lucid',
        codebook_size=16384,
        num_codebooks=1,
        latent_dim=32,
        norm_codes=True,
        norm_latents=False,
        prediction_type='sample',
        num_train_timesteps=1000,
        cls_free_guidance_dropout=0.2,
        masked_cfg=True,
        masked_cfg_low=0,
        masked_cfg_high=None,
        beta_schedule='linear',
        thresholding=True,
        sync_codebook=False,
        ema_decay=0.99,
        threshold_ema_dead_code=0.35,
        code_replacement_policy='batch_random',
        commitment_weight=1.0,
        kmeans_init=False,
        undo_std=False,
        freeze_enc=False,
        conditioning='concat',
        dec_transformer_dropout=0.2,
        zero_terminal_snr=True,
        init_encoder_from_vivit=False
    ).cuda()
    x = torch.rand(4, 3, 32, 224, 224).cuda()
    noised = torch.rand(4, 3, 32, 224, 224).cuda()
    y, loss = model(x, noised, torch.tensor([3,2,54,5]).cuda())
    print(y.shape)
    print(loss)
    print(str(model))
    """

    model = VQVAE(
        enc_type='HandTransformer',
        dec_type='HandTransformer',
        init_encoder=False,
        post_mlp=True,
        quant_type='lucid',
        codebook_size=1024,
        num_codebooks=1,
        latent_dim=32,
        norm_codes=True,
        norm_latents=False,
        sync_codebook=False,
        ema_decay=0.99,
        threshold_ema_dead_code=0.35,
        code_replacement_policy='batch_random',
        commitment_weight=1.0,
        kmeans_init=False,
        undo_std=False,
        freeze_enc=False,
        out_conv=False,
    ).cuda()
    x = torch.rand(1, 80, 21, 3).cuda()
    y, loss = model(x)
    print(y.shape)
    print(loss)
    print(str(model))
