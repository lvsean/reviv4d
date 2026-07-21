import numpy as np
import os
import torch
from torch.utils.data import Dataset
import glob
import tarfile
import io
import json
import pickle
# Data root, override with the REVIV_DATA_ROOT env var (default: ./example_data).
DATA_ROOT = os.environ.get("REVIV_DATA_ROOT", "./example_data")

NYMERIA_512_DATASET_ROOT = f'{DATA_ROOT}'


def _nymeria_512_data_files():
    return sorted(glob.glob(os.path.join(NYMERIA_512_DATASET_ROOT, '*', '*', 'data_file_gaze.pkl')))


def _is_nymeria_512_source(path):
    if path is None:
        return False
    return 'nymeria_512' in path or os.path.abspath(path) == NYMERIA_512_DATASET_ROOT


class CamTrajDataset(Dataset):
    # canonicalize camera trajectory with the first pose
    def __init__(self, mode='train', clip_len=60, args=None):
        
        self.data_path = args.data_path
        self.mode = mode
        self.clip_len = clip_len # num of frames. cut videos into clips
        assert self.clip_len == 60 # changed to 60 frames 2s.
        self.args = args
        
        self.dataset_samples = []
        self.mean = np.load(f'{DATA_ROOT}/cam/cam_mean_251031.npy')
        self.std = np.load(f'{DATA_ROOT}/cam/cam_std_251031.npy')
        
        if mode == 'train':
            #     # TODO: change this coord system to aria world coord is not needed due to canonicalization

            
            self.dataset_samples = np.load(f'{DATA_ROOT}/cam/cam_train_251031.npy')
            print(self.dataset_samples.shape)
            print('load cam_train.npy')
            print("load {} training samples".format(self.dataset_samples.shape[0]))
        elif mode == 'val':
            self.dataset_samples = np.load(f'{DATA_ROOT}/cam/cam_val_251031.npy')
            print("load cam_val.npy")
            print("load {} validation samples".format(self.dataset_samples.shape[0]))
        elif mode == 'tokenize':
            #         # Iterate over each file in the tar archive
            #             # Check if the file is a .npz file within the 'label/' directory
            #                 # Extract the file into memory
                            
            #                     # Load the .npz file using numpy
            #                         # Check if 'cam' is in the .npz file
            #                             # Access the 'cam' data
                                        
            #         # Iterate over each file in the tar archive
            #             # Check if the file is a .npz file within the 'label/' directory
            #                 # Extract the file into memory
                            
            
                
            if 'h2o' in args.tokenize_path or 'arctic' in args.tokenize_path or 'taco' in args.tokenize_path or 'hot3d' in args.tokenize_path:
                with tarfile.open(args.tokenize_path, 'r') as tar:
                    # Iterate over each file in the tar archive
                    for member in tar.getmembers():
                        # Check if the file is a .npz file within the 'label/' directory
                        if member.isfile() and member.name.endswith('.npz'):
                            # Extract the file into memory
                            file_obj = tar.extractfile(member)
                            
                            if file_obj is not None:
                                cam_data = np.load(io.BytesIO(file_obj.read()))['cam']
                                if np.any(np.isnan(cam_data)):
                                    continue
                                self.dataset_samples.append({'x': self.canonicalize(cam_data[:self.clip_len]), 'name': os.path.basename(member.name).split('.')[0] + '-0'})
                                self.dataset_samples.append({'x': self.canonicalize(cam_data[self.clip_len:]), 'name': os.path.basename(member.name).split('.')[0] + '-1'})
            elif 'egogen' in args.tokenize_path:
                
                print('Loading egogen data for tokenization...')

                opengl_to_opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

                with tarfile.open(args.tokenize_path, 'r') as tar:
                    # Iterate over each file in the tar archive
                    for member in tar.getmembers():
                        # Check if the file is a .npz file within the 'label/' directory
                        if member.isfile() and member.name.endswith('.npz'):
                            # Extract the file into memory
                            file_obj = tar.extractfile(member)
                            
                            if file_obj is not None:
                                cam_data = np.load(io.BytesIO(file_obj.read()))['arr_0']
                                if np.any(np.isnan(cam_data)):
                                    continue
                                self.dataset_samples.append({'x': self.canonicalize(cam_data[:self.clip_len] @ opengl_to_opencv), 'name': os.path.basename(member.name).split('.')[0] + '-0'})
                                self.dataset_samples.append({'x': self.canonicalize(cam_data[self.clip_len:] @ opengl_to_opencv), 'name': os.path.basename(member.name).split('.')[0] + '-1'})
            elif 'egoexo' in args.tokenize_path:
                
                print('Loading egoexo data for tokenization...')

                cw90 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

                with tarfile.open(args.tokenize_path, 'r') as tar:
                    # Iterate over each file in the tar archive
                    for member in tar.getmembers():
                        # Check if the file is a .npz file within the 'label/' directory
                        if member.isfile() and member.name.endswith('.npz'):
                            # Extract the file into memory
                            file_obj = tar.extractfile(member)
                            
                            if file_obj is not None:
                                cam_data = np.load(io.BytesIO(file_obj.read()))['cam']
                                if np.any(np.isnan(cam_data)):
                                    continue
                                self.dataset_samples.append({'x': self.canonicalize(cam_data[:self.clip_len] @ cw90), 'name': os.path.basename(member.name).split('.')[0] + '-0'})
                                self.dataset_samples.append({'x': self.canonicalize(cam_data[self.clip_len:] @ cw90), 'name': os.path.basename(member.name).split('.')[0] + '-1'})
            elif 'holoassist' in args.tokenize_path:
                
                print('Loading holoassist data for tokenization...')

                holocam2opencv = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

                with tarfile.open(args.tokenize_path, 'r') as tar:
                    # Iterate over each file in the tar archive
                    for member in tar.getmembers():
                        # Check if the file is a .npz file within the 'label/' directory
                        if member.isfile() and member.name.endswith('.npy'):
                            # Extract the file into memory
                            file_obj = tar.extractfile(member)
                            
                            if file_obj is not None:
                                cam_data = np.load(io.BytesIO(file_obj.read()))
                                if np.any(np.isnan(cam_data)):
                                    continue
                                self.dataset_samples.append({'x': self.canonicalize(cam_data[:self.clip_len] @ holocam2opencv), 'name': os.path.basename(member.name).split('.')[0] + '-0'})
                                self.dataset_samples.append({'x': self.canonicalize(cam_data[self.clip_len:] @ holocam2opencv), 'name': os.path.basename(member.name).split('.')[0] + '-1'})
            elif _is_nymeria_512_source(args.tokenize_path):

                print(f'Loading nymeria_512 camera data for tokenization from {NYMERIA_512_DATASET_ROOT}')
                data_files = _nymeria_512_data_files()
                if not data_files:
                    raise FileNotFoundError(f'No data_file_gaze.pkl files found under {NYMERIA_512_DATASET_ROOT}')

                for i, data_path in enumerate(data_files):
                    print(f'Processing {i + 1}/{len(data_files)}: {data_path}')
                    print(f'Number of samples so far: {len(self.dataset_samples)}')
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)

                    for item in data:
                        if 'extrinsics' not in item or 'id' not in item:
                            continue
                        cam = item['extrinsics']
                        if cam.shape[0] < self.clip_len or np.any(np.isnan(cam)):
                            continue

                        clip_idx = 0
                        for start in range(0, cam.shape[0] - self.clip_len + 1, self.clip_len):
                            cam_seq = cam[start:start + self.clip_len]
                            input_seq = np.linalg.inv(cam_seq)
                            converted = self.canonicalize(input_seq)
                            self.dataset_samples.append({'x': converted, 'name': f'{item["id"]}_{clip_idx}'})
                            clip_idx += 1

                print(f'Total {len(self.dataset_samples)} samples for tokenization from nymeria_512 dataset.')
            elif 'nymeria' in args.tokenize_path:

                print('Loading nymeria data for tokenization...')
                dataset_root_dir = f'{DATA_ROOT}/dataset/'
                seq_file = os.path.join(dataset_root_dir, 'train_val_split_251031.json')
                with open(seq_file, 'r') as f:
                    seq_data = json.load(f)
                seq_folders = seq_data['train'] + seq_data['val']

                for i in range(len(seq_folders)):
                    print(f'Processing {i+1}/{len(seq_folders)}: {seq_folders[i]}')
                    print(f'Number of samples so far: {len(self.dataset_samples)}')
                    folder_path = seq_folders[i]
                    data_path = os.path.join(folder_path, 'data_file_gaze.pkl')
                    if not os.path.exists(data_path):
                        raise FileNotFoundError(f'Data file not found: {data_path}')

                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    for item in data:
                        cam = item['extrinsics']
                        input_seq_0 = np.linalg.inv(cam[:self.clip_len])
                        input_seq_1 = np.linalg.inv(cam[self.clip_len:])
                        converted_0 = self.canonicalize(input_seq_0)
                        converted_1 = self.canonicalize(input_seq_1)
                        self.dataset_samples.append({'x': converted_0, 'name': f'{item["id"]}_0'})
                        self.dataset_samples.append({'x': converted_1, 'name': f'{item["id"]}_1'})

                print(f'Total {len(self.dataset_samples)} samples for tokenization from nymeria dataset.')
            elif 'adt' in args.tokenize_path:
                print('Loading adt data for tokenization...')
                path = f"{DATA_ROOT}/ADT/adt_2s_fps30/cam/cam_for_tokenize_adt.npy"
                with open(path, 'rb') as f:
                    train_data = np.load(f, allow_pickle=True).item()
                data = train_data['data']
                name = train_data['name']
                self.dataset_samples = [{'x': data[i], 'name': name[i]} for i in range(data.shape[0])]
                print(f"Loaded {data.shape} samples with {len(name)} from {args.tokenize_path}")
                print(f"Sample shape: {data.shape}")
                print(f"Name shape: {len(name)}")


            




    
    def read_pose(self, data_path):
        img_pose_array = []
        with open(data_path) as f:
            lines = f.read().split('\n')
            for line in lines:
                if line == '':  # end of the lines.
                    break
                line_data = list(map(float, line.split('\t')))
                img_pose_array.append(line_data)
            img_pose_array = np.array(img_pose_array)[:, 2:].reshape(-1, 4, 4)
        return img_pose_array
    
    def canonicalize(self, sample):
        # output: canonicalized 9d camera
        inv = np.linalg.inv(sample[0])
        canoed = np.einsum('ij, kjl -> kil', inv, sample)
        rot6d = canoed[:, :3, :2]
        transl = canoed[:, :3, 3:]
        cam_9d = np.concatenate((rot6d, transl), axis=-1).transpose(0, 2, 1).reshape(-1, 9)
        return cam_9d

    def __getitem__(self, index):
        if self.mode == 'train':
            sample = self.dataset_samples[index]
            return torch.tensor((sample - self.mean) / self.std).float()
        elif self.mode == 'val':
            sample = self.dataset_samples[index]
            if (self.mean == 0).all() or (self.std == 1).all(): # mean should be set in run_.py
                pass

            return torch.tensor((sample - self.mean) / self.std).float()
        elif self.mode == 'tokenize':
            if (self.mean == 0).all() or (self.std == 1).all():
                pass

            data = self.dataset_samples[index]
            return {'x': torch.tensor((data['x'] - self.mean) / self.std).float(), 'name': data['name']}
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)

if __name__ == '__main__':
    x = CamTrajDataset()
    pass

