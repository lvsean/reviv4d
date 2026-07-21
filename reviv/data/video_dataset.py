import os
import io
import numpy as np
import torch
from torchvision import transforms
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from reviv.data.random_erasing import RandomErasing
from reviv.data.video_transforms import (
    Compose, Resize, CenterCrop, Normalize,
    create_random_augment, random_short_side_scale_jitter, 
    random_crop, random_resized_crop_with_shift, random_resized_crop,
    horizontal_flip, random_short_side_scale_jitter, uniform_crop, horizontal_flip_np, 
    random_scaling, random_rotation, adjust_brightness, random_rotation_dep, add_gaussian_noise
)
from reviv.data.volume_transforms import ClipToTensor
import glob
import random
from reviv.utils.data_constants import (IMAGENET_DEFAULT_MEAN,
                                  IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN,
                                  IMAGENET_SURFACE_NORMAL_STD, IMAGENET_SURFACE_NORMAL_MEAN,
                                  IMAGENET_INCEPTION_STD, SEG_IGNORE_INDEX, PAD_MASK_VALUE)

# Data root, override with the REVIV_DATA_ROOT env var (default: ./example_data).
DATA_ROOT = os.environ.get("REVIV_DATA_ROOT", "./example_data")

try:
    from petrel_client.client import Client
    has_client = True
except ImportError:
    has_client = False

class VideoDataset(Dataset):
    """Load your own video dataset."""

    def __init__(self, mode='train', clip_len=32,
                 frame_sample_rate=1, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3, args=None):
        self.prefix = None
        self.mode = mode
        self.clip_len = clip_len # num of frames. cut videos into clips
        self.frame_sample_rate = frame_sample_rate # frame sampling rate (interval between two sampled frames).
        self.crop_size = args.crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment # Number of segments to evenly divide the video into clips.
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        assert num_segment == 1 # TODO: why only sample one clip during training? discard the whole loaded video?
        if self.mode in ['train']:
            self.aug = True
            # TODO
            self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")
        
        self.dataset_samples = []
        if mode == 'train':
            # holoassist
            with open(f'{DATA_ROOT}/datasets_aligned/rgb/holoassist/train_samples.txt', 'r') as f:
                train_list = f.read().splitlines()
            for x in train_list:
                self.dataset_samples.append(os.path.join(f'{DATA_ROOT}/datasets_aligned/rgb/holoassist/rgb_256_8fps/', x))
            # egoexo4d
            with open(f'{DATA_ROOT}/datasets_aligned/egoexo/train_samples.txt', 'r') as f:
                train_list = f.read().splitlines()
            for x in train_list:
                self.dataset_samples.append(os.path.join(f'{DATA_ROOT}/datasets_aligned/egoexo/rgb_256_8fps', x))

            # egogen
            for file in glob.glob(f'{DATA_ROOT}/egogen/rgb_256_8fps/*.mp4'):
                self.dataset_samples.append(file)

            # HOT3D
            with open(f'{DATA_ROOT}/datasets_aligned/rgb/hot3d/train_files.txt', 'r') as f:
                train_list = f.read().splitlines()
            for x in train_list:
                self.dataset_samples.append(os.path.join(f'{DATA_ROOT}/datasets_aligned/rgb/hot3d/rgb_256_8fps', x))

            # ARCTIC
            with open(f'{DATA_ROOT}/datasets_aligned/rgb/arctic/train_files.txt', 'r') as f:
                train_list = f.read().splitlines()
            for x in train_list:
                self.dataset_samples.append(os.path.join(f'{DATA_ROOT}/datasets_aligned/rgb/arctic/rgb_256_8fps', x))

            # TACO
            with open(f'{DATA_ROOT}/datasets_aligned/rgb/taco/train_files.txt', 'r') as f:
                train_list = f.read().splitlines()
            for x in train_list:
                self.dataset_samples.append(os.path.join(f'{DATA_ROOT}/datasets_aligned/rgb/taco/rgb_256_8fps', x))

            # H2O
            with open(f'{DATA_ROOT}/datasets_aligned/rgb/h2o/train_files.txt', 'r') as f:
                train_list = f.read().splitlines()
            for x in train_list:
                self.dataset_samples.append(os.path.join(f'{DATA_ROOT}/datasets_aligned/rgb/h2o/rgb_256_8fps', x))

            # Reinterhand
            with open(f'{DATA_ROOT}/datasets_aligned/rgb/reinterhand/train_files.txt', 'r') as f:
                train_list = f.read().splitlines()
            for x in train_list:
                self.dataset_samples.append(os.path.join(f'{DATA_ROOT}/datasets_aligned/rgb/reinterhand/rgb_256_8fps', x))
                
            print('train samples: ', len(self.dataset_samples))
                
        elif mode == 'val':
            # holoassist
            with open(f'{DATA_ROOT}/datasets_aligned/rgb/holoassist/val_samples.txt', 'r') as f:
                val_list = f.read().splitlines()
            for x in val_list:
                self.dataset_samples.append(os.path.join(f'{DATA_ROOT}/datasets_aligned/rgb/holoassist/rgb_256_8fps/', x))

            # egoexo4d
            with open(f'{DATA_ROOT}/datasets_aligned/egoexo/val_samples.txt', 'r') as f:
                val_list = f.read().splitlines()
            for x in val_list:
                self.dataset_samples.append(os.path.join(f'{DATA_ROOT}/datasets_aligned/egoexo/rgb_256_8fps', x))
            
            # HOT3D
            with open(f'{DATA_ROOT}/datasets_aligned/rgb/hot3d/val_files.txt', 'r') as f:
                val_list = f.read().splitlines()
            for x in val_list:
                self.dataset_samples.append(os.path.join(f'{DATA_ROOT}/datasets_aligned/rgb/hot3d/rgb_256_8fps', x))

            # ARCTIC
            with open(f'{DATA_ROOT}/datasets_aligned/rgb/arctic/val_files.txt', 'r') as f:
                val_list = f.read().splitlines()
            for x in val_list:
                self.dataset_samples.append(os.path.join(f'{DATA_ROOT}/datasets_aligned/rgb/arctic/rgb_256_8fps', x))

            # TACO
            with open(f'{DATA_ROOT}/datasets_aligned/rgb/taco/val_files.txt', 'r') as f:
                val_list = f.read().splitlines()
            for x in val_list:
                self.dataset_samples.append(os.path.join(f'{DATA_ROOT}/datasets_aligned/rgb/taco/rgb_256_8fps', x))

            # H2O
            with open(f'{DATA_ROOT}/datasets_aligned/rgb/h2o/val_files.txt', 'r') as f:
                val_list = f.read().splitlines()
            for x in val_list:
                self.dataset_samples.append(os.path.join(f'{DATA_ROOT}/datasets_aligned/rgb/h2o/rgb_256_8fps', x))

            # Reinterhand
            with open(f'{DATA_ROOT}/datasets_aligned/rgb/reinterhand/val_files.txt', 'r') as f:
                val_list = f.read().splitlines()
            for x in val_list:
                self.dataset_samples.append(os.path.join(f'{DATA_ROOT}/datasets_aligned/rgb/reinterhand/rgb_256_8fps', x))

            print('val samples: ', len(self.dataset_samples))
            
        elif mode == 'test':
            with open(os.path.join(self.prefix, 'data_split', 'test-v1_2.txt')) as f:
                test_list = f.read().splitlines()
            for x in test_list:
                for clip in glob.glob(os.path.join(self.prefix, 'video_pitch_shifted', x, 'Export_py', 'clip_6s', '*.mp4')):
                    self.dataset_samples.append(clip)
        elif mode == 'tokenize':
            for file in glob.glob(os.path.join(args.tokenize_path, '*.mp4')):
                self.dataset_samples.append(file)

        self.client = None
        if has_client:
            self.client = Client('~/petreloss.conf')
        
        pixel_mean = IMAGENET_INCEPTION_MEAN if not args.imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        pixel_std = IMAGENET_INCEPTION_STD if not args.imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        if (mode == 'train'):
            self.data_transform = Compose([
                CenterCrop(size=(self.crop_size, self.crop_size)), # T H W C -> T H' W' C
                ClipToTensor(), # T H W C -> C T H W
                Normalize(mean=pixel_mean, std=pixel_std)
            ])

        elif (mode == 'val' or mode == 'tokenize'):
            self.data_transform = Compose([
                CenterCrop(size=(self.crop_size, self.crop_size)),
                ClipToTensor(),
                Normalize(mean=pixel_mean, std=pixel_std)
            ])
        elif mode == 'test':
            pass

            self.data_resize = Compose([
                Resize(size=(short_side_size), interpolation='area')
            ])
            self.data_transform = Compose([
                ClipToTensor(),
                Normalize(mean=pixel_mean, std=pixel_std)
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args 
            scale_t = 1
            sample = self.dataset_samples[index]
            if args.use_npy:
                buffer = np.load(sample)
            else:
                buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t) # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)

            if args.num_aug_sample > 1:
                frame_list = []
                index_list = []
                for _ in range(args.num_aug_sample):
                    new_frames = self._aug_frame(buffer, args)
                    frame_list.append(new_frames)
                    index_list.append(index)
                return frame_list, index_list, {}
            elif args.num_aug_sample == 0:
                buffer = self.data_transform(buffer) # do not do random data augmentation now
            else:
                if args.domain == 'rgb':
                    # do data augmentation
                    if random.random() < 0.3:
                        buffer = horizontal_flip_np(buffer)
                    if random.random() < 0.3:
                        buffer = random_rotation(buffer)
                    if random.random() < 0.3:
                        buffer = adjust_brightness(buffer)
                    if random.random() < 0.3:
                        buffer = random_scaling(buffer)
                        
                    
                buffer = self.data_transform(buffer)
            
            return buffer # C T H W

        elif self.mode == 'val':
            sample = self.dataset_samples[index]
            if self.args.use_npy:
                buffer = np.load(sample)
            else:
                buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.data_transform(buffer)
            return buffer # , self.label_array[index], sample.split("/")[-1].split(".")[0]
        
        elif self.mode == 'tokenize':
            sample = self.dataset_samples[index]
            name = os.path.basename(sample).split('.')[0]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.data_transform(buffer)
            return {'x': torch.tensor(buffer).float(), 'name': name}
        
        elif self.mode == 'test':
            pass

            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample, chunk_nb=chunk_nb)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample, chunk_nb=chunk_nb)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            if self.test_num_crop == 1:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) / 2
                spatial_start = int(spatial_step)
            else:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                    / (self.test_num_crop - 1)
                spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[:, spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[:, :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                   chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
        self,
        buffer,
        args,
    ):
        pass

        aug_transform = create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C 
        
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        # [C T H W] -> [C T crop_size crop_size]
        buffer = spatial_sampling( 
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer


    def loadvideo_decord(self, sample, sample_rate_scale=1, chunk_nb=0):
        """Load video content using Decord"""
        fname = sample

        try:
            if self.keep_aspect_ratio:
                if "s3://" in fname:
                    video_bytes = self.client.get(fname)
                    vr = VideoReader(io.BytesIO(video_bytes),
                                     num_threads=1,
                                     ctx=cpu(0))
                else:
                    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                if "s3://" in fname:
                    video_bytes = self.client.get(fname)
                    vr = VideoReader(io.BytesIO(video_bytes),
                                     width=self.new_width,
                                     height=self.new_height,
                                     num_threads=1,
                                     ctx=cpu(0))
                else:
                    vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                    num_threads=1, ctx=cpu(0))

            # handle temporal segments
            converted_len = int(self.clip_len * self.frame_sample_rate)
            seg_len = len(vr) // self.num_segment

            if self.mode == 'test':
                temporal_step = max(1.0 * (len(vr) - converted_len) / (self.test_num_segment - 1), 0)
                temporal_start = int(chunk_nb * temporal_step)

                bound = min(temporal_start + converted_len, len(vr))
                all_index = [x for x in range(temporal_start, bound, self.frame_sample_rate)]
                while len(all_index) < self.clip_len:
                    all_index.append(all_index[-1])
                vr.seek(0)
                buffer = vr.get_batch(all_index).asnumpy()
                return buffer

            all_index = []
            for i in range(self.num_segment):
                if seg_len <= converted_len: # when video is shorter than the sampled clip. padding
                    index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
                    index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len))
                    index = np.clip(index, 0, seg_len - 1).astype(np.int64)
                else:
                    end_idx = np.random.randint(converted_len, seg_len)
                    str_idx = end_idx - converted_len
                    index = np.linspace(str_idx, end_idx, num=self.clip_len)
                    index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
                index = index + i*seg_len
                all_index.extend(list(index))

            all_index = all_index[::int(sample_rate_scale)]
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = random_crop(frames, crop_size)
        else:
            transform_func = (
                random_resized_crop_with_shift
                if motion_shift
                else random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

if __name__ == '__main__':
    x = VideoDataset(mode='train')
    y = VideoDataset(mode='val')