import numpy as np
import os
import torch
from torch.utils.data import Dataset
import glob
import tarfile
import io
import json
import pickle
import tqdm
# Data root, override with the REVIV_DATA_ROOT env var (default: ./example_data).
DATA_ROOT = os.environ.get("REVIV_DATA_ROOT", "./example_data")

NYMERIA_512_DATASET_ROOT = f'{DATA_ROOT}'


def _nymeria_512_data_files():
    return sorted(glob.glob(os.path.join(NYMERIA_512_DATASET_ROOT, '*', '*', 'data_file_gaze.pkl')))


def _is_nymeria_512_source(path):
    if path is None:
        return False
    return 'nymeria_512' in path or os.path.abspath(path) == NYMERIA_512_DATASET_ROOT


class GazeDataset(Dataset):
    def __init__(self, mode='train', clip_len=60, args=None):

        self.data_path = args.data_path
        self.mode = mode
        self.clip_len = clip_len # num of frames. cut videos into clips
        self.args = args
        
        self.dataset_samples = []
        self.sample_counts = {}
        self.mean = (0.5, 0.5) # [0, 1] -> [-1, 1]
        self.std = (0.5, 0.5)

        if mode == 'train':
            # load npy data directly. data processing is same as following code.
            self.dataset_samples = np.load(f'{DATA_ROOT}/gaze/gaze_train_60.npy')
            print(f"Loaded {len(self.dataset_samples)} training samples from npy file.")



            # # egoexo4d

            # # HOT3D

        elif mode == 'val':
            # load npy data directly. data processing is same as following code.
            self.dataset_samples = np.load(f'{DATA_ROOT}/gaze/gaze_val_60.npy')
            print(f"Loaded {len(self.dataset_samples)} validation samples from npy file.")

            
            # # egoexo4d
            
            # # HOT3D
        
        elif mode == 'val_plot':
            with open(os.path.join(self.data_path, 'data_split', 'val-v1_2.txt')) as f:
                val_list = f.read().splitlines()
            for x in val_list:
                data_path = os.path.join(self.data_path, x, 'Export_py', 'Eyes', 'Eyes_proj.txt')
                gaze_data = self.read_gaze_txt(data_path)
                for kk in range(0, gaze_data.shape[0] - self.clip_len + 1, 10):
                    sample = self.convert(gaze_data[kk : kk + self.clip_len], orig_res=[896, 504], resize_res=[896, 504])
                    self.dataset_samples.append({'vid': os.path.join(self.data_path, x, 'Export_py', 'Video_pitchshift.mp4'), 'x': sample, 'start_frame': kk, 'end_frame': kk + self.clip_len})

        elif mode == 'tokenize':
                    
            #         # Iterate over each file in the tar archive
            #             # Check if the file is a .npz file within the 'label/' directory
            #                 # Extract the file into memory
                            
            #                     # Load the .npz file using numpy
                                        
            

            if _is_nymeria_512_source(args.tokenize_path):
                self.sample_counts['nymeria_512'] = 0

                data_files = _nymeria_512_data_files()
                if not data_files:
                    raise FileNotFoundError(f'No data_file_gaze.pkl files found under {NYMERIA_512_DATASET_ROOT}')

                print(f'Loading nymeria_512 gaze data for tokenization from {NYMERIA_512_DATASET_ROOT}')
                for i, data_path in enumerate(data_files):
                    print(f'Processing {i + 1}/{len(data_files)}: {data_path}')
                    print(f'Number of samples so far: {self.sample_counts["nymeria_512"]}')
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)

                    for item in data:
                        if 'gaze' not in item or 'id' not in item:
                            continue
                        gaze = item['gaze']
                        if gaze.shape[0] < self.clip_len:
                            continue
                        converted = self.convert(gaze, orig_res=[1408, 1408], resize_res=[512, 512], new_res=[512, 512])

                        clip_idx = 0
                        for start in range(0, converted.shape[0] - self.clip_len + 1, self.clip_len):
                            self.dataset_samples.append({
                                'x': converted[start:start + self.clip_len],
                                'name': f'{item["id"]}_{clip_idx}',
                            })
                            self.sample_counts['nymeria_512'] += 1
                            clip_idx += 1

            elif 'nymeria' in args.tokenize_path:
                # Nymeria
                self.sample_counts['nymeria'] = 0

                dataset_root_dir = f'{DATA_ROOT}/dataset/'
                seq_file = os.path.join(dataset_root_dir, 'train_val_split_251031.json')
                with open(seq_file, 'r') as f:
                    seq_data = json.load(f)
                seq_folders = seq_data['train'] + seq_data['val']

                for i in range(len(seq_folders)):
                    print(f'Processing {i+1}/{len(seq_folders)}: {seq_folders[i]}')
                    print(f'Number of samples so far: {self.sample_counts["nymeria"]}')
                    folder_path = seq_folders[i]
                    data_path = os.path.join(folder_path, 'data_file_gaze.pkl')
                    if not os.path.exists(data_path):
                        raise FileNotFoundError(f'Data file not found: {data_path}')

                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    for item in data:
                        gaze = item['gaze']  # Assuming this is a numpy array of shape (T, 2)
                        converted = self.convert(gaze, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408])
                        self.dataset_samples.append({'x': converted[:self.clip_len], 'name': f'{item["id"]}_0'})
                        self.dataset_samples.append({'x': converted[self.clip_len:], 'name': f'{item["id"]}_1'})
                        self.sample_counts['nymeria'] += 2


            if 'holo' in args.tokenize_path:
                # HoloAssist
                self.sample_counts['holo'] = 0
                with open(f'{DATA_ROOT}/main_model_token/file_list/train_files_holoassist_no_dup.txt') as f:
                    train_list = f.read().splitlines()
                with open(f'{DATA_ROOT}/main_model_token/file_list/val_files_holoassist_no_dup.txt') as f:
                    val_list = f.read().splitlines()

                train_list = train_list + val_list
                for x in tqdm.tqdm(train_list):
                    npz_name = f"{DATA_ROOT}/holoassist/gaze_raw/holoassist/{x}.npy"
                    gaze_data = np.load(npz_name)
                    self.dataset_samples.append({'x': self.convert(gaze_data[:self.clip_len], orig_res=[896, 504], resize_res=[896, 504]), 'name': x + '-0'})
                    self.dataset_samples.append({'x': self.convert(gaze_data[self.clip_len:], orig_res=[896, 504], resize_res=[896, 504]), 'name': x + '-1'})
                    self.sample_counts['holo'] += 2

            if 'egoexo' in args.tokenize_path:
                # EgoExo4D
                self.sample_counts['egoexo'] = 0
                with open(f'{DATA_ROOT}/main_model_token/file_list/train_files_egoexo_no_dup.txt') as f:
                    train_list = f.read().splitlines()
                with open(f'{DATA_ROOT}/main_model_token/file_list/val_files_egoexo_no_dup.txt') as f:
                    val_list = f.read().splitlines()
                train_list = train_list + val_list
                for mp4_name in tqdm.tqdm(train_list):
                    npz_name = f"{DATA_ROOT}/egoexo/egoexo_cam_gaze/{mp4_name}.npz"
                    gaze_data = np.load(npz_name)['gaze']

                    converted = self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408])
                    self.dataset_samples.append({'x': converted[:self.clip_len], 'name': mp4_name + '-0'})
                    self.dataset_samples.append({'x': converted[self.clip_len:], 'name': mp4_name + '-1'})
                    self.sample_counts['egoexo'] += 2

            if 'hot3d' in args.tokenize_path:
                # HOT3D
                self.sample_counts['hot3d'] = 0
                with open(f'{DATA_ROOT}/main_model_token/file_list/train_files_hot3d_no_dup.txt') as f:
                    train_list = f.read().splitlines()
                with open(f'{DATA_ROOT}/main_model_token/file_list/val_files_hot3d_no_dup.txt') as f:
                    val_list = f.read().splitlines()
                train_list = train_list + val_list
                for mp4_name in tqdm.tqdm(train_list):
                    npz_name = f"{DATA_ROOT}/hot3d/processed_data/{mp4_name}.npz"
                    gaze_data = np.load(npz_name)['gaze']
                    converted = self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408])
                    self.dataset_samples.append({'x': converted[:self.clip_len], 'name': mp4_name + '-0'})
                    self.dataset_samples.append({'x': converted[self.clip_len:], 'name': mp4_name + '-1'})
                    self.sample_counts['hot3d'] += 2
            
            print("Sample counts by dataset:", self.sample_counts)


    def read_gaze_txt(self, gaze_path):
        with open(gaze_path) as f:
            gaze_data = []
            lines = f.read().split('\n')
            for line in lines:
                if line == '':  # end of the lines.
                    break
                line_data = list(map(float, line.strip().split('\t')))
                gaze_data.append(line_data)
            gaze_data = np.array(gaze_data)[:, 2:]
        return gaze_data
    
    def convert(self, gaze_data, orig_res, resize_res, new_res=[480, 480]):
        # convert gaze 2d coordinates in the original resolution to 480x480
        orig_res = np.array(orig_res)
        new_res = np.array(new_res)
        gaze_normed = gaze_data / orig_res # to [0, 1]
        gaze_resize_coord = gaze_normed * np.array(resize_res) # resized coord

        # check if valid in center cropped image (new_res)
        _min = (resize_res - new_res) / 2
        gaze_new_coord = gaze_resize_coord - _min
        gaze = gaze_new_coord / np.array(new_res)

        mask = np.ones(gaze.shape[0]) # invalid val: 0
        nan = np.where(np.isnan(gaze).any(-1))[0]
        mask[nan] = 0
        gaze[nan] = 0. # inpute nan with 0.

        # many noise in the GT data, filter out gaze very outside of the image
        out = np.where((gaze > 1.2).any(-1))[0]
        mask[out] = 0
        gaze[out] = 0.
        out = np.where((gaze < -0.2).any(-1))[0]
        mask[out] = 0
        gaze[out] = 0

        gaze = (gaze - self.mean) / self.std # normalize data to -1,1
        return np.concatenate([gaze, mask.reshape(-1, 1)], axis=-1)

    def __getitem__(self, index):
        if self.mode == 'val_plot':
            data = self.dataset_samples[index]
            return {'vid': data['vid'], 
                    'x': torch.tensor(data['x']).float(),
                    'start_frame': data['start_frame'],
                    'end_frame': data['end_frame']}
        elif self.mode == 'tokenize':
            data = self.dataset_samples[index]
            return {'x':torch.tensor(data['x']).float(), 'name': data['name']}
        else:
            return torch.tensor(self.dataset_samples[index]).float()

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)

if __name__ == '__main__':
    x = GazeDataset()
    pass

