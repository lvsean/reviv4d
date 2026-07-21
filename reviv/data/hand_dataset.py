import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
from reviv.utils.data_constants import (ARCTIC_LHAND_MEAN, ARCTIC_RHAND_MEAN, \
                                        ARCTIC_LHAND_STD, ARCTIC_RHAND_STD)



# Root directory for processed clip data. Override with REVIV_DATA_ROOT env var.
DATA_ROOT = os.environ.get("REVIV_DATA_ROOT", "./example_data")


class HandDataset(Dataset):
    """Load your own video dataset."""

    def __init__(self, mode='train', clip_len=30, args=None):
        
        self.data_path = args.data_path
        if 'lhand' not in self.data_path and 'rhand' not in self.data_path:
            pass


        self.mode = mode
        self.clip_len = clip_len # num of frames. cut videos into clips
        self.args = args
        
        self.dataset_samples = []
        if mode == 'train':
            for sid in [1, 2, 4, 5, 6, 7, 9, 10]:
                for clip in glob.glob(os.path.join(self.data_path, "s%02d*" % sid)):
                    self.dataset_samples.append(clip)
        elif mode == 'val': # use s08 as validation set
            for clip in glob.glob(os.path.join(self.data_path, "s08*")):
                self.dataset_samples.append(clip)
        elif mode == 'test':
            pass

        
        self.hand_mean = ARCTIC_LHAND_MEAN if 'lhand' in self.data_path else ARCTIC_RHAND_MEAN
        self.hand_std = ARCTIC_LHAND_STD if 'lhand' in self.data_path else ARCTIC_RHAND_STD

    def __data_transform(self, x):
        # x: hand motions TxVx3
        tensor = torch.from_numpy(x)
        dtype = tensor.dtype
        mean = torch.as_tensor(self.hand_mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.hand_std, dtype=dtype, device=tensor.device)
        tensor.sub_(mean[None, None, :]).div_(std[None, None, :])
        return tensor

    def __getitem__(self, index):
        if self.mode == 'train':
            sample = self.dataset_samples[index]
            buffer = np.load(sample) # TxVx3
            # TODO: maybe do some augmentation here
            buffer = self.__data_transform(buffer)
            return buffer 
        elif self.mode == 'val':
            sample = self.dataset_samples[index]
            buffer = np.load(sample)
            buffer = self.__data_transform(buffer)
            return buffer # , self.label_array[index], sample.split("/")[-1].split(".")[0]
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)
