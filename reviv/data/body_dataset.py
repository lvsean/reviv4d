import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import pickle
import os

# Root directory for processed clip data. Override with REVIV_DATA_ROOT env var.
DATA_ROOT = os.environ.get("REVIV_DATA_ROOT", "./example_data")
NYMERIA_512_DATASET_ROOT = DATA_ROOT


def _nymeria_512_data_files():
    return sorted(glob.glob(os.path.join(NYMERIA_512_DATASET_ROOT, '*', '*', 'data_file_gaze.pkl')))


def _is_nymeria_512_source(path):
    if path is None:
        return False
    return 'nymeria_512' in path or os.path.abspath(path) == NYMERIA_512_DATASET_ROOT


def _rotation_from_zyx_without_yaw(rotation):
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])

    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)

    rot_y = np.array([
        [cos_pitch, 0, sin_pitch],
        [0, 1, 0],
        [-sin_pitch, 0, cos_pitch],
    ])
    rot_x = np.array([
        [1, 0, 0],
        [0, cos_roll, -sin_roll],
        [0, sin_roll, cos_roll],
    ])
    return rot_y @ rot_x


def _canonicalize_nymeria_body(joints_w, w2c):
    batch_size, joint_num, _ = joints_w.shape
    joints_w = np.transpose(joints_w, (2, 0, 1))
    joints_w = joints_w.reshape(3, -1)
    c2w = np.linalg.inv(w2c)

    cam_r = w2c[0, :3, :3].reshape(3, 3)
    cam_t = w2c[0, :3, 3].reshape(3, 1)
    joints_cam = cam_r @ joints_w + cam_t

    cam_r = _rotation_from_zyx_without_yaw(c2w[0, :3, :3].reshape(3, 3))
    joints_cam = cam_r @ joints_cam

    joints_cam = joints_cam.reshape(3, batch_size, joint_num)
    joints_cam = np.transpose(joints_cam, (1, 2, 0))
    return joints_cam


def _load_nymeria_512_body_for_tokenization(clip_len):
    keep_list = [i for i in range(22)]
    keep_list.remove(3)

    data = []
    name = []
    data_files = _nymeria_512_data_files()
    if not data_files:
        raise FileNotFoundError(f'No data_file_gaze.pkl files found under {NYMERIA_512_DATASET_ROOT}')

    print(f'Loading nymeria_512 body data for tokenization from {NYMERIA_512_DATASET_ROOT}')
    for file_idx, data_path in enumerate(data_files):
        print(f'Processing {file_idx + 1}/{len(data_files)}: {data_path}')
        with open(data_path, 'rb') as f:
            items = pickle.load(f)

        for item in items:
            if 'joints_w' not in item or 'extrinsics' not in item or 'id' not in item:
                continue
            joints_w = item['joints_w'][:, keep_list, :]
            cam = item['extrinsics']
            if joints_w.shape[0] < clip_len or cam.shape[0] < clip_len:
                continue
            if np.any(np.isnan(joints_w)) or np.any(np.isnan(cam)):
                continue

            max_frames = min(joints_w.shape[0], cam.shape[0])
            clip_idx = 0
            for start in range(0, max_frames - clip_len + 1, clip_len):
                end = start + clip_len
                data.append(_canonicalize_nymeria_body(joints_w[start:end], cam[start:end]))
                name.append(f'{item["id"]}_{clip_idx}')
                clip_idx += 1

    return torch.from_numpy(np.array(data)).float(), name


class BodyDataset(Dataset):
    """Load your own video dataset."""

    def __init__(self, mode='train', clip_len=30, args=None):
        
        self.data_path = args.data_path

        joints_num, cood = args.data_path.split('_')[0], args.data_path.split('_')[1]

        self.mode = mode
        self.clip_len = clip_len # num of frames. cut videos into clips
        self.args = args
        
        self.dataset_samples = []


        if "body_nymeria_train_60" in self.data_path:
            mean_path = f"{DATA_ROOT}/body/body_nymeria_mean_60.npy"
            std_path = f"{DATA_ROOT}/body/body_nymeria_std_60.npy"
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)    
            print(f"Loaded mean from {mean_path}, std from {std_path}, shape: {self.mean.shape}, {self.std.shape}")

            if mode == 'train':
                file_path = f"{DATA_ROOT}/body/body_nymeria_train_60.npy"
                print(f"Loading training data from {file_path}")

                self.dataset_samples = np.load(file_path) # N x T x V x 3
                self.dataset_samples = self.dataset_samples - self.mean
                self.dataset_samples = self.dataset_samples / self.std
                self.dataset_samples = torch.from_numpy(self.dataset_samples).float()
                print(f"Loaded {self.dataset_samples.shape[0]} samples from {file_path}")

            elif mode == 'val': # use s08 as validation set
                file_path = f"{DATA_ROOT}/body/body_nymeria_val_60.npy"
                print(f"Loading validation data from {file_path}")
                self.dataset_samples = np.load(file_path) # N x T x V x 3

                self.dataset_samples = self.dataset_samples - self.mean
                self.dataset_samples = self.dataset_samples / self.std
                self.dataset_samples = torch.from_numpy(self.dataset_samples).float()
                print(f"Loaded {self.dataset_samples.shape[0]} samples from {file_path}")
                print(f"Data shape: {self.dataset_samples.shape}, mean shape: {self.mean.shape}, std shape: {self.std.shape}")
            
            elif mode == 'tokenize' and _is_nymeria_512_source(getattr(args, 'tokenize_path', None)):
                self.dataset_samples, self.name = _load_nymeria_512_body_for_tokenization(self.clip_len)
                print(f'Loaded {self.dataset_samples.shape[0]} nymeria_512 body samples for tokenization')
                print(f'Data shape: {self.dataset_samples.shape}, mean shape: {self.mean.shape}, std shape: {self.std.shape}')

                self.dataset_samples = self.dataset_samples - self.mean
                self.dataset_samples = self.dataset_samples / self.std
                self.dataset_samples = self.dataset_samples.float()

            elif mode == 'tokenize':
                file_path = f"{DATA_ROOT}/body/train_21_ego_251031_tokenize.pkl"
                with open(file_path, 'rb') as f:
                    train_data = pickle.load(f)
                file_path = f"{DATA_ROOT}/body/val_21_ego_251031_tokenize.pkl"
                with open(file_path, 'rb') as f:
                    val_data = pickle.load(f)


                data = []
                name = []

                print(train_data['data'][0].shape)
                print(train_data['name'][0])

                for i in range(len(train_data['data'])):
                    data.append(train_data['data'][i])
                    name.append(train_data['name'][i])
                
                for i in range(len(val_data['data'])):
                    data.append(val_data['data'][i])
                    name.append(val_data['name'][i])

                self.dataset_samples = torch.from_numpy(np.array(data)).float()
                self.name = name
                print(f"Loaded {self.dataset_samples.shape[0]} samples with {len(self.name)} from {file_path}")
                print(f"Data shape: {self.dataset_samples.shape}, mean shape: {self.mean.shape}, std shape: {self.std.shape}")
                
                self.dataset_samples = self.dataset_samples - self.mean
                self.dataset_samples = self.dataset_samples / self.std
                self.dataset_samples = self.dataset_samples.float()

            else:
                raise ValueError(f"Unsupported mode {mode} for body_nymeria dataset")
        else:
            mean_path = f"{DATA_ROOT}/body/body_mean_21_ego_251031.npy"
            std_path = f"{DATA_ROOT}/body/body_std_21_ego_251031.npy"
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)    
            print(f"Loaded mean from {mean_path}, std from {std_path}")

            if mode == 'train':

                file_path = f"{DATA_ROOT}/body/train_21_ego_251031.npy"
                print(f"Loading training data from {file_path}")
                self.dataset_samples = np.load(file_path) # N x T x V x 3


                self.dataset_samples = self.dataset_samples - self.mean
                self.dataset_samples = self.dataset_samples / self.std

                

                self.dataset_samples = torch.from_numpy(self.dataset_samples).float()
                print(f"Loaded {self.dataset_samples.shape[0]} samples from {file_path}")
                print(f"Data shape: {self.dataset_samples.shape}, mean shape: {self.mean.shape}, std shape: {self.std.shape}")

            elif mode == 'val': # use s08 as validation set
                file_path = f"{DATA_ROOT}/body/val_21_ego_251031.npy"
                print(f"Loading validation data from {file_path}")
                self.dataset_samples = np.load(file_path) # N x T x V x 3

                self.dataset_samples = self.dataset_samples - self.mean
                self.dataset_samples = self.dataset_samples / self.std
                self.dataset_samples = torch.from_numpy(self.dataset_samples).float()
                print(f"Loaded {self.dataset_samples.shape[0]} samples from {file_path}")
                print(f"Data shape: {self.dataset_samples.shape}, mean shape: {self.mean.shape}, std shape: {self.std.shape}")
            
            elif mode == 'tokenize' and _is_nymeria_512_source(getattr(args, 'tokenize_path', None)):
                self.dataset_samples, self.name = _load_nymeria_512_body_for_tokenization(self.clip_len)
                print(f'Loaded {self.dataset_samples.shape[0]} nymeria_512 body samples for tokenization')
                print(f'Data shape: {self.dataset_samples.shape}, mean shape: {self.mean.shape}, std shape: {self.std.shape}')

                self.dataset_samples = self.dataset_samples - self.mean
                self.dataset_samples = self.dataset_samples / self.std
                self.dataset_samples = self.dataset_samples.float()

            elif mode == 'tokenize':
                file_path = f"{DATA_ROOT}/body/train_21_ego_251031_tokenize.pkl"
                with open(file_path, 'rb') as f:
                    train_data = pickle.load(f)
                file_path = f"{DATA_ROOT}/body/val_21_ego_251031_tokenize.pkl"
                with open(file_path, 'rb') as f:
                    val_data = pickle.load(f)

                data = []
                name = []

                print(train_data['data'][0].shape)
                print(train_data['name'][0])

                for i in range(len(train_data['data'])):
                    data.append(train_data['data'][i])
                    name.append(train_data['name'][i])
                
                for i in range(len(val_data['data'])):
                    data.append(val_data['data'][i])
                    name.append(val_data['name'][i])




                self.dataset_samples = torch.from_numpy(np.array(data)).float()
                self.name = name
                print(f"Loaded {self.dataset_samples.shape[0]} samples with {len(self.name)} from {file_path}")
                print(f"Data shape: {self.dataset_samples.shape}, mean shape: {self.mean.shape}, std shape: {self.std.shape}")
                
                self.dataset_samples = self.dataset_samples - self.mean
                self.dataset_samples = self.dataset_samples / self.std
                self.dataset_samples = self.dataset_samples.float()
            
            elif mode == 'test':
                raise NotImplementedError("'test' mode is not supported for BodyDataset")
            

    #     # x: hand motions TxVx3

    def __getitem__(self, index):

        ## use the temporal augmentation in training
        if self.mode == "train":
            return self.dataset_samples[index]
            sample_range = self.dataset_samples[index].shape[0] - self.clip_len
            start_idx = np.random.randint(0, sample_range)


            return self.dataset_samples[index, start_idx:start_idx+self.clip_len, :, :]
        
        elif self.mode == "tokenize":
            return {
                'x': self.dataset_samples[index, :, :, :],
                'name': self.name[index]
            }
        
        return self.dataset_samples[index]
        
    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)

if __name__ == '__main__':
    dataset = BodyDataset(mode='val', clip_len=60, args=None)
    print(len(dataset))
    sample = dataset[1000]
    sample = sample * dataset.std + dataset.mean
    print(sample[0].shape)
    save_path = "../body_recon.pkl"
    save_dict = {
        'data': sample[0].numpy(),
        'mean': dataset.mean,
        'std': dataset.std
    }
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)
