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
from functools import partial

import reviv.utils.data_constants as data_constants
from reviv.data.modality_transforms import (CaptionTransform, DepthTransform,
                                      DetectionTransform, MaskTransform,
                                      NormalTransform, RGBTransform, RGBVideoTransform,
                                      SemsegTransform, TokTransform,
                                      CaptionEmbTransform, MetadataTransform,
                                      HumanPoseTransform, ColorPaletteTransform,
                                      SAMInstanceTokTransform, SAMInstanceTransform)
from reviv.models.decoder_embeddings import (ImageTokenDecoderEmbedding, VideoTokenDecoderEmbedding, HandTokenDecoderEmbedding,
                                      BodyTokenDecoderEmbedding,
                                             GazeCamTokenDecoderEmbedding, SequenceDecoderEmbedding)
from reviv.models.encoder_embeddings import (ImageEncoderEmbedding,
                                   ImageTokenEncoderEmbedding,
                                   SequenceEncoderEmbedding,
                                   SequenceEmbEncoderEmbedding,
                                   VideoTokenEncoderEmbedding,
                                   VideoEncoderEmbedding,
                                   GazeCamTokenEncoderEmbedding,
                                   HandTokenEncoderEmbedding,
                                      BodyTokenEncoderEmbedding,
                                   )
from reviv.utils import generate_uint15_hash

MODALITY_INFO = {
    # ReViV-7 modalities
    'rgb@256': {
        'input_size': 256,
        'patch_size': 8,
        'encoder_embedding': partial(VideoEncoderEmbedding, num_channels=3, patch_size=(4, 8, 8), num_frames=16),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': 4096,
        'type': 'img',
        'num_channels': 3,
        'id': generate_uint15_hash('rgb@256'),
        'path': 'rgb_video',
        'tensor_shape': (16, 256, 256, 3),
    },
    'rgb512': {
        'input_size': 512,
        'patch_size': 8,
        'encoder_embedding': partial(VideoEncoderEmbedding, num_channels=3, patch_size=(8, 16, 16), image_size=512, num_frames=32),
        'decoder_embedding': None,
        'min_tokens': 0,
        # 32 frames with patch size (8, 16, 16) at 512x512 -> 4 * 32 * 32 tokens.
        'max_tokens': 4096,
        'type': 'img',
        'num_channels': 3,
        'id': generate_uint15_hash('rgb512'),
        'path': 'rgb_video_512',
        'tensor_shape': (32, 512, 512, 3),
    },
    'rgb': { # used for tokenizer training
        'type': 'img',
        'num_channels': 3,
        'id': generate_uint15_hash('rgb'),
        'path': 'rgb',
    },
    'lhand': {
        'type': 'mesh',
        'num_channels': 3,
        'id': generate_uint15_hash('lhand'),
    },
    'rhand': {
        'type': 'mesh',
        'num_channels': 3,
        'id': generate_uint15_hash('rhand'),
    },
    'body': {
        'type': 'keypoints',
        'num_channels': 3,
        'id': generate_uint15_hash('body'),
    },
    'fullbody': {
        'type': 'keypoints',
        'num_channels': 3,
        'id': generate_uint15_hash('fullbody'),
    },
    'motion': {
        'type': 'keypoints',
        'num_channels': 3,
        'id': generate_uint15_hash('motion'),
    },
    'lhand_joint': {
        'type': 'keypoints',
        'num_channels': 3,
        'id': generate_uint15_hash('lhand_joint'),
    },
    'rhand_joint': {
        'type': 'keypoints',
        'num_channels': 3,
        'id': generate_uint15_hash('rhand_joint'),
    },
    'cam': {
        'type': 'cam',
        'num_channels': 9,
        'id': generate_uint15_hash('cam'),
    },
    'tok_body': {
        'vocab_size': 2048,
        'encoder_embedding': partial(BodyTokenEncoderEmbedding, vocab_size=2048), # TODO
        'decoder_embedding': partial(BodyTokenDecoderEmbedding, vocab_size=2048),
        'min_tokens': 0,
        'max_tokens': 210, 
        'type': 'img',
        'id': generate_uint15_hash('tok_body'),
        'pretokenized': True,
        'path': 'body'
    },
    'tok_lhand': {
        'vocab_size': 1024,
        'encoder_embedding': partial(HandTokenEncoderEmbedding, vocab_size=1024), # TODO
        'decoder_embedding': partial(HandTokenDecoderEmbedding, vocab_size=1024),
        'min_tokens': 0,
        'max_tokens': 210, 
        'type': 'img',
        'id': generate_uint15_hash('tok_lhand'),
        'pretokenized': True,
        'path': 'lhand'
    },
    'tok_rhand': {
        'vocab_size': 1024,
        'encoder_embedding': partial(HandTokenEncoderEmbedding, vocab_size=1024), # TODO
        'decoder_embedding': partial(HandTokenDecoderEmbedding, vocab_size=1024),
        'min_tokens': 0,
        'max_tokens': 210, 
        'type': 'img',
        'id': generate_uint15_hash('tok_rhand'),
        'pretokenized': True,
        'path': 'rhand'
    },
    'tok_cam': {
        'vocab_size': 256,
        'encoder_embedding': partial(GazeCamTokenEncoderEmbedding, vocab_size=256), # TODO
        'decoder_embedding': partial(GazeCamTokenDecoderEmbedding, vocab_size=256),
        'min_tokens': 0,
        'max_tokens': 30, 
        'type': 'cam',
        'id': generate_uint15_hash('tok_cam'),
        'pretokenized': True,
        'path': 'cam'
    },
    'gaze': {
        'type': 'gaze',
        'num_channels': 2,
        'id': generate_uint15_hash('gaze'),
    },
    'tok_gaze': {
        'vocab_size': 512,
        'encoder_embedding': partial(GazeCamTokenEncoderEmbedding, vocab_size=512), # TODO
        'decoder_embedding': partial(GazeCamTokenDecoderEmbedding, vocab_size=512),
        'min_tokens': 0,
        'max_tokens': 30, 
        'type': 'gaze',
        'id': generate_uint15_hash('tok_gaze'),
        'pretokenized': True,
        'path': 'gaze'
    },
    'caption': {
        'vocab_size': 30_000,
        'encoder_embedding': partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        'decoder_embedding': partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        'min_tokens': 0,
        'max_tokens': 256,
        'type': 'seq',
        'id': generate_uint15_hash('caption'),
    },
    'tok_caption': {
        'vocab_size': 30_000,
        'encoder_embedding': partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        'decoder_embedding': partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        'min_tokens': 0,
        'max_tokens': 256,
        'type': 'seq',
        'id': generate_uint15_hash('tok_caption'),
        'pretokenized': True,
        'path': 'text',
    },
    'det': {
        'vocab_size': 30_000,
        'encoder_embedding': partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        'decoder_embedding': partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        'min_tokens': 0,
        'max_tokens': 256,
        'type': 'seq',
        'id': generate_uint15_hash('det'),
    },
    'tok_rgb@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 16384,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=16384),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=16384),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_rgb@224'),
        'pretokenized': True,
    },
    'tok_rgb': { # for rgb video
        'input_size': 256,
        'patch_size': 8,
        'vocab_size': 64000,
        'encoder_embedding': partial(VideoTokenEncoderEmbedding, vocab_size=64000, patch_size=(4, 8, 8), image_size=256, token_shape=(5, 32, 32)),
        'decoder_embedding': partial(VideoTokenDecoderEmbedding, vocab_size=64000, patch_size=(4, 8, 8), image_size=256, token_shape=(5, 32, 32)),
        'min_tokens': 0,
        'max_tokens': 5120, 
        'type': 'img',
        'id': generate_uint15_hash('tok_rgb'),
        'pretokenized': True,
        'path': 'rgb',
        'token_shape': (5, 32, 32),
    },
    'tok_depth': { # for rgb video
        'input_size': 256,
        'patch_size': 8,
        'vocab_size': 64000,
        'encoder_embedding': partial(VideoTokenEncoderEmbedding, vocab_size=64000, patch_size=(4, 8, 8), image_size=256, token_shape=(5, 32, 32)),
        'decoder_embedding': partial(VideoTokenDecoderEmbedding, vocab_size=64000, patch_size=(4, 8, 8), image_size=256, token_shape=(5, 32, 32)),
        'min_tokens': 0,
        'max_tokens': 5120, 
        'type': 'img',
        'id': generate_uint15_hash('tok_depth'),
        'pretokenized': True,
        'path': 'depth',
        'token_shape': (5, 32, 32),
    },
    'tok_rgb_512': {
        'vocab_size': 64000,
        'encoder_embedding': partial(VideoTokenEncoderEmbedding, vocab_size=64000, patch_size=(4, 16, 16), image_size=512, token_shape=(5, 32, 32)),
        'decoder_embedding': partial(VideoTokenDecoderEmbedding, vocab_size=64000, patch_size=(4, 16, 16), image_size=512, token_shape=(5, 32, 32)),
        'min_tokens': 0,
        'max_tokens': 5120,
        'type': 'img',
        'id': generate_uint15_hash('tok_rgb_512'),
        'pretokenized': True,
        'path': 'rgb_512',
        'token_shape': (5, 32, 32),
    },
    'tok_depth_512': {
        'vocab_size': 64000,
        'encoder_embedding': partial(VideoTokenEncoderEmbedding, vocab_size=64000, patch_size=(4, 16, 16), image_size=512, token_shape=(5, 32, 32)),
        'decoder_embedding': partial(VideoTokenDecoderEmbedding, vocab_size=64000, patch_size=(4, 16, 16), image_size=512, token_shape=(5, 32, 32)),
        'min_tokens': 0,
        'max_tokens': 5120,
        'type': 'img',
        'id': generate_uint15_hash('tok_depth_512'),
        'pretokenized': True,
        'path': 'depth_512',
        'token_shape': (5, 32, 32),
    },
    'tok_depth@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_depth@224'),
        'pretokenized': True,
    },
    'depth': { # used for tokenizer training
        'type': 'img',
        'num_channels': 1,
        'id': generate_uint15_hash('depth'),
    },
    'tok_normal@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_normal@224'),
        'pretokenized': True,
    },
    'normal': { # used for tokenizer training
        'type': 'img',
        'num_channels': 3,
        'id': generate_uint15_hash('normal'),
    },
    'tok_semseg@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 4096,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=4096),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=4096),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_semseg@224'),
        'pretokenized': True,
    },
    'semseg_coco': { # used for tokenizer training
        'type': 'img', 
        'num_channels': 64,
        'num_labels': data_constants.COCO_SEMSEG_NUM_CLASSES,
        'id': generate_uint15_hash('semseg_coco'),
    },
    'tok_clip@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_clip@224'),
        'pretokenized': True,
    },
    'CLIP-B16': { # used for tokenizer training
        'type': 'feature_map',
        'num_channels': 512,
        'id': generate_uint15_hash('CLIP-B16'),
    },

    # ReViV-21 modalities
    't5_caption': {
        'encoder_embedding': partial(SequenceEmbEncoderEmbedding, max_length=77, padding_idx=0),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': 77,
        'type': 'seq_emb',
        'id': generate_uint15_hash('t5_caption'),
    },
    'metadata': { 
        'vocab_size': 30_000,
        'encoder_embedding': partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=40, padding_idx=0, sincos_pos_emb=True),
        'decoder_embedding': partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=40, padding_idx=0, sincos_pos_emb=True),
        'min_tokens': 0,
        'max_tokens': 40, # At most 2x19=38 for 19 metadata types, +1 for EOS, +1 for sentinel
        'type': 'seq',
        'id': generate_uint15_hash('metadata'),
        'shared_vocab': ['caption'],
        'path': 'metadata',
    },
    'human_poses': { 
        'vocab_size': 30_000,
        'encoder_embedding': partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=263, padding_idx=0, sincos_pos_emb=True),
        'decoder_embedding': partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=263, padding_idx=0, sincos_pos_emb=True),
        'min_tokens': 0,
        'max_tokens': 275, #7*39+1 EOS+1 S_1#263, #261 in one of the models, or 263 to have EOS #261+1+1 #238,
        'type': 'seq',
        'num_channels': 207, # for tokenization training, only the pose part is needed
        'id': generate_uint15_hash('human_poses'),
        'shared_vocab': ['caption'],
    },
    'color_palette': { 
        'vocab_size': 30_000,
        'encoder_embedding': partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=23, padding_idx=0, sincos_pos_emb=True),
        'decoder_embedding': partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=23, padding_idx=0, sincos_pos_emb=True),
        'min_tokens': 0,
        'max_tokens': 23, #7x3=21 for 7 colors, +1 for EOS, +1 for sentinel
        'type': 'seq',
        'id': generate_uint15_hash('color_palette'),
        'shared_vocab': ['caption'],
        'path': 'color_palette',
    },
    'sam_mask': {
        'encoder_embedding': None,
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': 64,
        'type': 'img',
        'num_channels': 1,
        'id': generate_uint15_hash('sam_mask'),
    },
    'sam_instance': {
        'vocab_size': 30_000,
        'encoder_embedding': partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=290, padding_idx=0, sincos_pos_emb=True),
        'decoder_embedding': partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=290, padding_idx=0, sincos_pos_emb=True),
        'min_tokens': 0,
        'max_tokens': 290,
        'type': 'seq',
        'id': generate_uint15_hash('sam_instance'),
        'shared_vocab': ['caption'],
        'pretokenized': True,
    },
    'tok_canny_edge@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_canny_edge@224'),
        'pretokenized': True,
    },
    'canny_edge': { # used for tokenizer training
        'type': 'img',
        'num_channels': 1,
        'id': generate_uint15_hash('canny_edge'),
    },
    'tok_sam_edge@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_sam_edge@224'),
        'pretokenized': True,
    },
    'tok_dinov2@224': {
        'input_size': 224,
        'patch_size': 14,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 256
        'type': 'img',
        'id': generate_uint15_hash('tok_dinov2@224'),
        'pretokenized': True,
    },
    'DINOv2-B14': { # used for tokenizer training
        'type': 'feature_map',
        'num_channels': 768,
        'id': generate_uint15_hash('DINOv2-B14'),
    },
    'tok_imagebind@224': {
        'input_size': 224,
        'patch_size': 14,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 256
        'type': 'img',
        'id': generate_uint15_hash('tok_imagebind@224'),
        'pretokenized': True,
    },
    'ImageBind-H14': { # used for tokenizer training
        'type': 'feature_map',
        'num_channels': 1280,
        'id': generate_uint15_hash('ImageBind-H14'),
    },
    'tok_dinov2_global': {
        'vocab_size': 8192,
        'patch_size': 56,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192, sincos_pos_emb=False),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192, sincos_pos_emb=False),
        'min_tokens': 0,
        'max_tokens': 16,
        'type': 'img',
        'id': generate_uint15_hash('tok_dinov2_global'),
        'pretokenized': True,
    },
    'DINOv2-B14-global': { # used for tokenizer training
        'type': 'feature_map',
        'num_channels': 768,
        'id': generate_uint15_hash('DINOv2-B14-global'),
    },
    'tok_imagebind_global': {
        'vocab_size': 8192,
        'patch_size': 56,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192, sincos_pos_emb=False),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192, sincos_pos_emb=False),
        'min_tokens': 0,
        'max_tokens': 16,
        'type': 'img',
        'id': generate_uint15_hash('tok_imagebind_global'),
        'pretokenized': True,
    },
    'ImageBind-H14-global': { # used for tokenizer training
        'type': 'feature_map',
        'num_channels': 1280,
        'id': generate_uint15_hash('ImageBind-H14-global'),
    },

    ### 224->448 super resolution modalities
    'rgb@448': {
        'input_size': 448,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=3),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'num_channels': 3,
        'id': generate_uint15_hash('rgb@448'),
        'path': 'rgb',
    },
    'tok_rgb@448': {
        'input_size': 448,
        'patch_size': 16,
        'vocab_size': 16384,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=16384),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=16384),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'id': generate_uint15_hash('tok_rgb@448'),
        'pretokenized': True,
    },
    'tok_depth@448': {
        'input_size': 448,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'id': generate_uint15_hash('tok_depth@448'),
        'pretokenized': True,
    },
    'tok_normal@448': {
        'input_size': 448,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'id': generate_uint15_hash('tok_normal@448'),
        'pretokenized': True,
    },
    'tok_semseg@448': {
        'input_size': 448,
        'patch_size': 16,
        'vocab_size': 4096,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=4096),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=4096),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'id': generate_uint15_hash('tok_semseg@448'),
        'pretokenized': True,
    },
    'tok_clip@448': {
        'input_size': 448,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'id': generate_uint15_hash('tok_clip@448'),
        'pretokenized': True,
    },
}


def infer_token_vocab_sizes(state_dict):
    """Read each token modality's vocab size from a main-model state dict.

    Different checkpoints train the same modality with different codebook
    sizes (e.g. tok_body is 2048 in the reviv / metric_depth checkpoints but
    1024 in reviv_500b). The size is the first dim of a modality's decoder
    ``to_logits.weight`` (or, for input-only modalities, its encoder
    ``token_emb.weight``). Returns {modality: vocab_size}.
    """
    enc_prefix, enc_suffix = "encoder_embeddings.", ".token_emb.weight"
    dec_prefix, dec_suffix = "decoder_embeddings.", ".to_logits.weight"
    vocab_sizes = {}
    for key, tensor in state_dict.items():
        if key.startswith(dec_prefix) and key.endswith(dec_suffix):
            mod = key[len(dec_prefix): -len(dec_suffix)]
            vocab_sizes[mod] = int(tensor.shape[0])
        elif key.startswith(enc_prefix) and key.endswith(enc_suffix):
            mod = key[len(enc_prefix): -len(enc_suffix)]
            vocab_sizes.setdefault(mod, int(tensor.shape[0]))
    return vocab_sizes


def apply_vocab_overrides(modality_info, vocab_sizes):
    """Return a copy of modality_info with vocab sizes overridden.

    Rebuilds the encoder/decoder embedding partials so the embedding tables and
    output heads are created at the checkpoint's codebook size. Modalities whose
    size already matches (or that carry no size) are passed through untouched.
    """
    patched = {}
    for mod, info in modality_info.items():
        target = vocab_sizes.get(mod)
        if target is None or target == info.get("vocab_size"):
            patched[mod] = info
            continue
        new_info = dict(info)
        new_info["vocab_size"] = target
        for key in ("encoder_embedding", "decoder_embedding"):
            emb = info.get(key)
            if emb is not None and "vocab_size" in getattr(emb, "keywords", {}):
                new_info[key] = partial(
                    emb.func, *emb.args, **{**emb.keywords, "vocab_size": target}
                )
        patched[mod] = new_info
    return patched


# Note: @res suffix is ignored for modality transforms
MODALITY_TRANSFORMS = {
    # ReViV-7 modalities
    'rgb': RGBVideoTransform(), # RGBTransform(imagenet_default_mean_and_std=True),
    'rgb512': RGBVideoTransform(),
    # 'rgb@256': RGBVideoTransform(imagenet_default_mean_and_std=False),
    'caption': CaptionTransform(aligned_captions=True),
    'det': DetectionTransform(det_threshold=0.6, det_max_instances=None, bbox_order='dist_to_orig', coord_bins=1000, min_visibility=0.0),
    'tok_rgb': TokTransform(),
    'tok_rgb_512': TokTransform(),
    'tok_cam': TokTransform(),
    'tok_gaze': TokTransform(),
    'tok_depth': TokTransform(),
    'tok_depth_512': TokTransform(),
    'tok_lhand': TokTransform(),
    'tok_rhand': TokTransform(),
    'tok_body': TokTransform(),
    'tok_caption': TokTransform(),
    'tok_normal': TokTransform(),
    'tok_semseg': TokTransform(),
    'tok_clip': TokTransform(),
    # ReViV-21 modalities
    't5_caption': CaptionEmbTransform(),
    'metadata': MetadataTransform(special_vmin=0, special_vmax=999, shuffle=True, random_trunc=False, return_chunks=True),
    'human_poses': HumanPoseTransform(coord_bins=1000),
    'color_palette': ColorPaletteTransform(coord_bins=1000),
    'sam_instance': SAMInstanceTokTransform(image_size=224, points_per_side=7, point_order='random'),
    'tok_canny_edge': TokTransform(),
    'tok_sam_edge': TokTransform(),
    'tok_dinov2': TokTransform(),
    'tok_imagebind': TokTransform(),
    'tok_dinov2_global': TokTransform(),
    'tok_imagebind_global': TokTransform(),
    # Other
    'mask_valid': MaskTransform(mask_pool_size=1),
}

MODALITY_TRANSFORMS_DIVAE = {
    'rgb': RGBTransform(imagenet_default_mean_and_std=False),
    'depth': DepthTransform(standardize_depth=True),
    'normal': NormalTransform(standardize_surface_normals=False),
    'mask_valid': MaskTransform(mask_pool_size=1),
    'semseg_coco': SemsegTransform(shift_idx_by_one=True),
    'canny_edge': RGBTransform(imagenet_default_mean_and_std=False),
    'human_poses': HumanPoseTransform(coord_bins=1000, only_pose=True),
    'sam_mask': SAMInstanceTransform(mask_size=64, max_instance_n=1),
}

MODALITY_TRANSFORMS_VQCONTROLNET = {
    'rgb': RGBTransform(imagenet_default_mean_and_std=False),
    'mask_valid': MaskTransform(mask_pool_size=1),
    'caption': CaptionTransform(aligned_captions=True),
}
