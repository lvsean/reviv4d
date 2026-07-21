import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob



# Root directory for processed clip data. Override with REVIV_DATA_ROOT env var.
DATA_ROOT = os.environ.get("REVIV_DATA_ROOT", "./example_data")


class HandJointDataset(Dataset):
    """Load your own video dataset."""

    def __init__(self, mode='train', clip_len=60, args=None):
        

        self.mode = mode
        self.clip_len = clip_len # num of frames. cut videos into clips
        self.args = args

        source = getattr(args, 'hand_data_source', None) or 'default'  # 'default' | 'camspace'

        if source == 'camspace':

            if not args.rhand:
                self.mean = np.load(f'{DATA_ROOT}/hand/all_hand_mean_60_camspace_withholo.npy')
                self.std = np.load(f'{DATA_ROOT}/hand/all_hand_std_60_camspace_withholo.npy')
                print(f'################load {DATA_ROOT}/hand/all_hand_mean_60_camspace_withholo.npy')
            else:
                self.mean = np.load(f'{DATA_ROOT}/hand/all_hand_mean_60_camspace_withholo_rhand.npy')
                self.std = np.load(f'{DATA_ROOT}/hand/all_hand_std_60_camspace_withholo_rhand.npy')
                print(f'################load {DATA_ROOT}/hand/all_hand_mean_60_camspace_withholo_rhand.npy')
            self.dataset_samples = []
            if mode == 'train':
                if not args.rhand:
                    self.dataset_samples = np.load(f'{DATA_ROOT}/hand/all_hand_train_60_camspace_withholo.npy')
                    print(f'load {DATA_ROOT}/hand/all_hand_train_60_camspace_withholo.npy')
                else:
                    self.dataset_samples = np.load(f'{DATA_ROOT}/hand/all_hand_train_60_camspace_withholo_rhand.npy')
                    print(f'load {DATA_ROOT}/hand/all_hand_train_60_camspace_withholo_rhand.npy')
            elif mode == 'val':
                if not args.rhand:
                    self.dataset_samples = np.load(f'{DATA_ROOT}/hand/all_hand_val_60_camspace_withholo.npy')
                    print(f'load {DATA_ROOT}/hand/all_hand_val_60_camspace_withholo.npy')
                else:
                    self.dataset_samples = np.load(f'{DATA_ROOT}/hand/all_hand_val_60_camspace_withholo_rhand.npy')
                    print(f'load {DATA_ROOT}/hand/all_hand_val_60_camspace_withholo_rhand.npy')
            elif mode == 'test':
                pass

                
            elif mode == 'tokenize':
                if 'holoassist' in args.tokenize_path:
                    if 'lhand' in args.tokenize_save_path:
                        print('=================================')
                        print('LEFT HAND GEN!!!!!!!!!!!!!!!!!')
                        print('=================================')
                        for file in glob.glob(f'{DATA_ROOT}/holoassist/holoassist_aligned_hand_v2/*.npz'):
                            all_data = np.load(file)
                            if 'lhand' not in all_data.files:
                                continue
                            lhand = all_data['lhand']
                            lhand_visible = all_data['lhand_visible']
                            for idx in range(2):
                                hand_data = lhand[idx * self.clip_len : (idx + 1) * self.clip_len]
                                if (lhand_visible == False).all(): # no hand visible, use <pad> token
                                    np.savez_compressed(os.path.join(args.tokenize_save_path, 'token', os.path.basename(file).split('.')[0] + f'-{idx}'), np.full((30, 7), args.codebook_size)) # use max codebooksize as <pad>
                                else:
                                    self.dataset_samples.append({'x': self.canonicalize(hand_data), 'name': os.path.basename(file).split('.')[0] + f'-{idx}'})
                                
                    elif 'rhand' in args.tokenize_save_path:
                        print('+++++++++++++++++++++++++++++++++')
                        print('RIGHT HAND!!!!!!!!!!!!!!!!!')
                        print('+++++++++++++++++++++++++++++++++')
                        for file in glob.glob(f'{DATA_ROOT}/holoassist/holoassist_aligned_hand_v2/*.npz'):
                            all_data = np.load(file)
                            if 'rhand' not in all_data.files:
                                continue
                            rhand = all_data['rhand']
                            assert rhand.ndim == 3
                            rhand_visible = all_data['rhand_visible']
                            for idx in range(2):
                                hand_data = rhand[idx * self.clip_len : (idx + 1) * self.clip_len]
                                if (rhand_visible == False).all(): # no hand visible, use <pad> token
                                    np.savez_compressed(os.path.join(args.tokenize_save_path, 'token', os.path.basename(file).split('.')[0] + f'-{idx}'), np.full((30, 7), args.codebook_size)) # use max codebooksize as <pad>
                                else:
                                    self.dataset_samples.append({'x': self.canonicalize(hand_data), 'name': os.path.basename(file).split('.')[0] + f'-{idx}'})
                                
                elif 'hot3d' in args.tokenize_path:
                    if 'lhand' in args.tokenize_save_path:
                        print('=================================')
                        print('LEFT HAND!!!!!!!!!!!!!!!!!')
                        print('=================================')
                        for file in glob.glob(f'{DATA_ROOT}/hot3d/processed_data/*.npz'):
                            all_data = np.load(file)
                            if 'lhand' not in all_data.files:
                                continue
                            lhand = all_data['lhand']
                            for idx in range(2):
                                hand_data = lhand[idx * self.clip_len : (idx + 1) * self.clip_len]
                                if np.all(np.isnan(hand_data)):
                                    pass
                                else:
                                    self.dataset_samples.append({'x': self.canonicalize(hand_data), 'name': os.path.basename(file).split('.')[0] + f'-{idx}'})
                    elif 'rhand' in args.tokenize_save_path:
                        print('+++++++++++++++++++++++++++++++++')
                        print('RIGHT HAND!!!!!!!!!!!!!!!!!')
                        print('+++++++++++++++++++++++++++++++++')
                        for file in glob.glob(f'{DATA_ROOT}/hot3d/processed_data/*.npz'):
                            all_data = np.load(file)
                            if 'rhand' not in all_data.files:
                                continue
                            rhand = all_data['rhand']
                            assert rhand.ndim == 3
                            for idx in range(2):
                                hand_data = rhand[idx * self.clip_len : (idx + 1) * self.clip_len]
                                
                                if np.all(np.isnan(hand_data)):
                                    pass
                                else:
                                    self.dataset_samples.append({'x': self.canonicalize(hand_data), 'name': os.path.basename(file).split('.')[0] + f'-{idx}'})
                    else:
                        pass

                
                elif 'arctic' in args.tokenize_path:
                    if 'lhand' in args.tokenize_save_path:
                        print('=================================')
                        print('LEFT HAND!!!!!!!!!!!!!!!!!')
                        print('=================================')
                        for file in glob.glob(f'{DATA_ROOT}/arctic/arctic_hand_in_cam/arctic_label/*.npz'):
                            lhand = np.load(file)['lhand']
                            self.dataset_samples.append({'x': self.canonicalize(lhand[:self.clip_len]), 'name': os.path.basename(file).split('.')[0] + '-0'})
                            self.dataset_samples.append({'x': self.canonicalize(lhand[self.clip_len:]), 'name': os.path.basename(file).split('.')[0] + '-1'})
                    elif 'rhand' in args.tokenize_save_path:
                        print('+++++++++++++++++++++++++++++++++')
                        print('RIGHT HAND!!!!!!!!!!!!!!!!!')
                        print('+++++++++++++++++++++++++++++++++')
                        for file in glob.glob(f'{DATA_ROOT}/arctic/arctic_hand_in_cam/arctic_label/*.npz'):
                            rhand = np.load(file)['rhand']
                            assert rhand.ndim == 3
                            self.dataset_samples.append({'x': self.canonicalize(rhand[:self.clip_len]), 'name': os.path.basename(file).split('.')[0] + '-0'})
                            self.dataset_samples.append({'x': self.canonicalize(rhand[self.clip_len:]), 'name': os.path.basename(file).split('.')[0] + '-1'})
                
                elif 'taco' in args.tokenize_path:
                    if 'lhand' in args.tokenize_save_path:
                        print('=================================')
                        print('LEFT HAND!!!!!!!!!!!!!!!!!')
                        print('=================================')
                        for file in glob.glob(f'{DATA_ROOT}/taco/taco_label/*.npz'):
                            lhand = np.load(file)['lhand']
                            self.dataset_samples.append({'x': self.canonicalize(lhand[:self.clip_len]), 'name': os.path.basename(file).split('.')[0] + '-0'})
                            self.dataset_samples.append({'x': self.canonicalize(lhand[self.clip_len:]), 'name': os.path.basename(file).split('.')[0] + '-1'})
                    elif 'rhand' in args.tokenize_save_path:
                        print('+++++++++++++++++++++++++++++++++')
                        print('RIGHT HAND!!!!!!!!!!!!!!!!!')
                        print('+++++++++++++++++++++++++++++++++')
                        for file in glob.glob(f'{DATA_ROOT}/taco/taco_label/*.npz'):
                            rhand = np.load(file)['rhand']
                            assert rhand.ndim == 3
                            self.dataset_samples.append({'x': self.canonicalize(rhand[:self.clip_len]), 'name': os.path.basename(file).split('.')[0] + '-0'})
                            self.dataset_samples.append({'x': self.canonicalize(rhand[self.clip_len:]), 'name': os.path.basename(file).split('.')[0] + '-1'})
                
                elif 'h2o' in args.tokenize_path:
                    if 'lhand' in args.tokenize_save_path:
                        print('=================================')
                        print('LEFT HAND!!!!!!!!!!!!!!!!!')
                        print('=================================')
                        for file in glob.glob(f'{DATA_ROOT}/h2o/h2o_handincam/*.npz'):
                            lhand = np.load(file)['lhand']
                            self.dataset_samples.append({'x': self.canonicalize(lhand[:self.clip_len]), 'name': os.path.basename(file).split('.')[0] + '-0'})
                            self.dataset_samples.append({'x': self.canonicalize(lhand[self.clip_len:]), 'name': os.path.basename(file).split('.')[0] + '-1'})
                    elif 'rhand' in args.tokenize_save_path:
                        print('+++++++++++++++++++++++++++++++++')
                        print('RIGHT HAND!!!!!!!!!!!!!!!!!')
                        print('+++++++++++++++++++++++++++++++++')
                        for file in glob.glob(f'{DATA_ROOT}/h2o/h2o_handincam/*.npz'):
                            rhand = np.load(file)['rhand']
                            assert rhand.ndim == 3
                            self.dataset_samples.append({'x': self.canonicalize(rhand[:self.clip_len]), 'name': os.path.basename(file).split('.')[0] + '-0'})
                            self.dataset_samples.append({'x': self.canonicalize(rhand[self.clip_len:]), 'name': os.path.basename(file).split('.')[0] + '-1'})

            
        elif source == 'default':
            if not args.rhand:
                self.mean = np.load(f'{DATA_ROOT}/hand/hand_mean_60_lhand.npy')
                self.std = np.load(f'{DATA_ROOT}/hand/hand_std_60_lhand.npy')
                print(f'#######load {DATA_ROOT}/hand/hand_mean_60_lhand.npy')
            else:
                self.mean = np.load(f'{DATA_ROOT}/hand/hand_mean_60_rhand.npy')
                self.std = np.load(f'{DATA_ROOT}/hand/hand_std_60_rhand.npy')
                print(f'#######load {DATA_ROOT}/hand/hand_mean_60_rhand.npy')

            if mode == 'train':
                if not args.rhand:
                    self.dataset_samples = np.load(f'{DATA_ROOT}/hand/hand_train_60_lhand.npy')
                    print(f'load {DATA_ROOT}/hand/hand_train_60_lhand.npy')
                else:
                    self.dataset_samples = np.load(f'{DATA_ROOT}/hand/hand_train_60_rhand.npy')
                    print(f'load {DATA_ROOT}/hand/hand_train_60_rhand.npy')
            elif mode == 'val':
                if not args.rhand:
                    self.dataset_samples = np.load(f'{DATA_ROOT}/hand/hand_val_60_lhand.npy')
                    print(f'load {DATA_ROOT}/hand/hand_val_60_lhand.npy')
                else:
                    self.dataset_samples = np.load(f'{DATA_ROOT}/hand/hand_val_60_rhand.npy')
                    print(f'load {DATA_ROOT}/hand/hand_val_60_rhand.npy')

            elif mode == 'tokenize':
                if 'holoassist' in args.tokenize_path:
                    if 'lhand' in args.tokenize_save_path:
                        print('=================================')
                        print('LEFT HAND!!!!!!!!!!!!!!!!!')
                        print('=================================')
                        for file in glob.glob(f'{DATA_ROOT}/holoassist/holoassist_aligned_hand_v2/*.npz'):
                            all_data = np.load(file)
                            if 'lhand' not in all_data.files:
                                continue
                            lhand = all_data['lhand']
                            lhand_visible = all_data['lhand_visible']
                            for idx in range(2):
                                hand_data = lhand[idx * self.clip_len : (idx + 1) * self.clip_len]
                                if (lhand_visible == False).all(): # no hand visible, use <pad> token
                                    np.savez_compressed(os.path.join(args.tokenize_save_path, 'token', os.path.basename(file).split('.')[0] + f'-{idx}'), np.full((30, 7), args.codebook_size)) # use max codebooksize as <pad>
                                else:
                                    self.dataset_samples.append({'x': self.canonicalize(hand_data), 'name': os.path.basename(file).split('.')[0] + f'-{idx}'})
                                
                    elif 'rhand' in args.tokenize_save_path:
                        print('+++++++++++++++++++++++++++++++++')
                        print('RIGHT HAND!!!!!!!!!!!!!!!!!')
                        print('+++++++++++++++++++++++++++++++++')
                        for file in glob.glob(f'{DATA_ROOT}/holoassist/holoassist_aligned_hand_v2/*.npz'):
                            all_data = np.load(file)
                            if 'rhand' not in all_data.files:
                                continue
                            rhand = all_data['rhand']
                            assert rhand.ndim == 3
                            rhand_visible = all_data['rhand_visible']
                            for idx in range(2):
                                hand_data = rhand[idx * self.clip_len : (idx + 1) * self.clip_len]
                                if (rhand_visible == False).all(): # no hand visible, use <pad> token
                                    np.savez_compressed(os.path.join(args.tokenize_save_path, 'token', os.path.basename(file).split('.')[0] + f'-{idx}'), np.full((30, 7), args.codebook_size)) # use max codebooksize as <pad>
                                else:
                                    self.dataset_samples.append({'x': self.canonicalize(hand_data), 'name': os.path.basename(file).split('.')[0] + f'-{idx}'})
                                
                elif 'hot3d' in args.tokenize_path:
                    if 'lhand' in args.tokenize_save_path:
                        print('=================================')
                        print('LEFT HAND!!!!!!!!!!!!!!!!!')
                        print('=================================')
                        for file in glob.glob(f'{DATA_ROOT}/hot3d/processed_data/*.npz'):
                            all_data = np.load(file)
                            if 'lhand' not in all_data.files:
                                continue
                            lhand = all_data['lhand']
                            for idx in range(2):
                                hand_data = lhand[idx * self.clip_len : (idx + 1) * self.clip_len]
                                if np.all(np.isnan(hand_data)):
                                    pass
                                else:
                                    self.dataset_samples.append({'x': self.canonicalize(hand_data), 'name': os.path.basename(file).split('.')[0] + f'-{idx}'})
                    elif 'rhand' in args.tokenize_save_path:
                        print('+++++++++++++++++++++++++++++++++')
                        print('RIGHT HAND!!!!!!!!!!!!!!!!!')
                        print('+++++++++++++++++++++++++++++++++')
                        for file in glob.glob(f'{DATA_ROOT}/hot3d/processed_data/*.npz'):
                            all_data = np.load(file)
                            if 'rhand' not in all_data.files:
                                continue
                            rhand = all_data['rhand']
                            assert rhand.ndim == 3
                            for idx in range(2):
                                hand_data = rhand[idx * self.clip_len : (idx + 1) * self.clip_len]
                                
                                if np.all(np.isnan(hand_data)):
                                    pass
                                else:
                                    self.dataset_samples.append({'x': self.canonicalize(hand_data), 'name': os.path.basename(file).split('.')[0] + f'-{idx}'})
                    else:
                        pass

                
                elif 'arctic' in args.tokenize_path:
                    if 'lhand' in args.tokenize_save_path:
                        print('=================================')
                        print('LEFT HAND!!!!!!!!!!!!!!!!!')
                        print('=================================')
                        for file in glob.glob(f'{DATA_ROOT}/arctic/arctic_hand_in_cam/arctic_label/*.npz'):
                            lhand = np.load(file)['lhand']
                            self.dataset_samples.append({'x': self.canonicalize(lhand[:self.clip_len]), 'name': os.path.basename(file).split('.')[0] + '-0'})
                            self.dataset_samples.append({'x': self.canonicalize(lhand[self.clip_len:]), 'name': os.path.basename(file).split('.')[0] + '-1'})
                    elif 'rhand' in args.tokenize_save_path:
                        print('+++++++++++++++++++++++++++++++++')
                        print('RIGHT HAND!!!!!!!!!!!!!!!!!')
                        print('+++++++++++++++++++++++++++++++++')
                        for file in glob.glob(f'{DATA_ROOT}/arctic/arctic_hand_in_cam/arctic_label/*.npz'):
                            rhand = np.load(file)['rhand']
                            assert rhand.ndim == 3
                            self.dataset_samples.append({'x': self.canonicalize(rhand[:self.clip_len]), 'name': os.path.basename(file).split('.')[0] + '-0'})
                            self.dataset_samples.append({'x': self.canonicalize(rhand[self.clip_len:]), 'name': os.path.basename(file).split('.')[0] + '-1'})
                
                elif 'taco' in args.tokenize_path:
                    if 'lhand' in args.tokenize_save_path:
                        print('=================================')
                        print('LEFT HAND!!!!!!!!!!!!!!!!!')
                        print('=================================')
                        for file in glob.glob(f'{DATA_ROOT}/taco/taco_label/*.npz'):
                            lhand = np.load(file)['lhand']
                            self.dataset_samples.append({'x': self.canonicalize(lhand[:self.clip_len]), 'name': os.path.basename(file).split('.')[0] + '-0'})
                            self.dataset_samples.append({'x': self.canonicalize(lhand[self.clip_len:]), 'name': os.path.basename(file).split('.')[0] + '-1'})
                    elif 'rhand' in args.tokenize_save_path:
                        print('+++++++++++++++++++++++++++++++++')
                        print('RIGHT HAND!!!!!!!!!!!!!!!!!')
                        print('+++++++++++++++++++++++++++++++++')
                        for file in glob.glob(f'{DATA_ROOT}/taco/taco_label/*.npz'):
                            rhand = np.load(file)['rhand']
                            assert rhand.ndim == 3
                            self.dataset_samples.append({'x': self.canonicalize(rhand[:self.clip_len]), 'name': os.path.basename(file).split('.')[0] + '-0'})
                            self.dataset_samples.append({'x': self.canonicalize(rhand[self.clip_len:]), 'name': os.path.basename(file).split('.')[0] + '-1'})
                
                elif 'h2o' in args.tokenize_path:
                    if 'lhand' in args.tokenize_save_path:
                        print('=================================')
                        print('LEFT HAND!!!!!!!!!!!!!!!!!')
                        print('=================================')
                        for file in glob.glob(f'{DATA_ROOT}/h2o/h2o_handincam/*.npz'):
                            lhand = np.load(file)['lhand']
                            self.dataset_samples.append({'x': self.canonicalize(lhand[:self.clip_len]), 'name': os.path.basename(file).split('.')[0] + '-0'})
                            self.dataset_samples.append({'x': self.canonicalize(lhand[self.clip_len:]), 'name': os.path.basename(file).split('.')[0] + '-1'})
                    elif 'rhand' in args.tokenize_save_path:
                        print('+++++++++++++++++++++++++++++++++')
                        print('RIGHT HAND!!!!!!!!!!!!!!!!!')
                        print('+++++++++++++++++++++++++++++++++')
                        for file in glob.glob(f'{DATA_ROOT}/h2o/h2o_handincam/*.npz'):
                            rhand = np.load(file)['rhand']
                            assert rhand.ndim == 3
                            self.dataset_samples.append({'x': self.canonicalize(rhand[:self.clip_len]), 'name': os.path.basename(file).split('.')[0] + '-0'})
                            self.dataset_samples.append({'x': self.canonicalize(rhand[self.clip_len:]), 'name': os.path.basename(file).split('.')[0] + '-1'})
                    
    #     # x: hand motions TxVx3
    
    #     # seq: 60 x 21 x 3 for a motion seq. idx 4: middle mcp. idx 0: wrist
    #     # Get first frame's wrist and MCP





    
    def canonicalize(self, seq): # no canonicalization in camera space
        return seq
    
    def impute_nan(self, seq):
        imputed = np.nan_to_num(seq, nan=0.)
        valid_mask = ~np.isnan(seq).any(-1)[:, :, None]
        imputed_with_mask = np.concatenate((imputed, valid_mask), -1)
        return imputed_with_mask

    def __getitem__(self, index):
        if self.mode == 'train':
            sample = self.dataset_samples[index]
            return torch.tensor(self.impute_nan((sample - self.mean) / self.std)).float()
        elif self.mode == 'val':
            sample = self.dataset_samples[index]
            return torch.tensor(self.impute_nan((sample - self.mean) / self.std)).float()
        elif self.mode == 'tokenize':
            sample = self.dataset_samples[index]
            return {'x':torch.tensor(self.impute_nan((sample['x'] - self.mean) / self.std)).float(), 'name': sample['name']}
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)
