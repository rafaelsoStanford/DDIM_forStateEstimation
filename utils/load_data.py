import numpy as np
import os
import pickle
import zarr
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

def visualize_batch(batch):
    print("Visualizing Batch and Data structure")
    print(" [ B, t_sequence, dims ]")
    for key, value in batch.items():
        print()
        print(f'--> Key: {key}')
        print(f'Shape: ({len(value)}, {len(value[0])})')
        print(value.shape)
        print("Min: ", value.min())
        print("Max: ", value.max())
        print()

    for traj in range(batch['position'].shape[0]):
        plt.plot(batch['position'][traj, :, 0], batch['position'][traj, :, 1])
        plt.scatter(0, 0, c='r')
        plt.waitforbuttonpress()
        plt.close()

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def save_stats(stats, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(stats, f)

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx # All index in one episode

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result



# =========== data set class  ===========
# loading data from zarr file
class CarRacingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 ):
        
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.sequence_len = obs_horizon + pred_horizon # chunk lenght of data
        self.normalized_train_data = {}

        print("Loading Data from Zarr File")

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')

        # float32, [0,1], (N,96,96,3)
        train_image_data = dataset_root['data']['img'][:]
        train_image_data = np.moveaxis(train_image_data, -1,1)


        # (N, D)
        train_data = {
            # Create Prediction Targets
            'position': dataset_root['data']['position'][:], # (T,2)
            'velocity': dataset_root['data']['velocity'][:], # (T,2)
            'action': dataset_root['data']['action'][:] #(T,3)
        }
        episode_ends = dataset_root['meta']['episode_ends'][:]


        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length= obs_horizon+pred_horizon,
            pad_before= 0,
            pad_after= 0)

        # ========== Normalize Data ============ 
        # normalized data to [-1,1], images are assumed to be normalized 
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])
        self.stats = stats

        self.normalized_train_data['image'] = train_image_data # [:,: , :80, :80] # Assumed to be already normalized, cropped to 80x80 removing the black border
        self.normalized_train_data['action'] = normalized_train_data['action'] # train_data['action'] # All action space values are constrained to [-1,1]
        self.normalized_train_data['velocity'] = normalized_train_data['velocity'] # train_data['velocity'] # All velocity values are constrained to [-1,1]
        self.normalized_train_data['position'] = train_data['position'] 
        self.indices = indices
        # self.stats = stats
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=    self.sequence_len,
            buffer_start_idx=   buffer_start_idx,
            buffer_end_idx=     buffer_end_idx,
            sample_start_idx=   sample_start_idx,
            sample_end_idx=     sample_end_idx
        )

        # # ========== normalize sample ============
        sample_stat = get_data_stats(nsample['position'])
        sample_normalized = normalize_data(nsample['position'], sample_stat)
        translation_vec = sample_normalized[0,:]
        nsample_centered = sample_normalized - translation_vec
        nsample['position'] = nsample_centered / 2.0

        return nsample

# =========== data loader module ===========
# data module
class CarRacingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size,  data_dir: str = "path/to/dir" , T_obs=4, T_pred=8 , T_act =1, seed=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.T_act = T_act

        self.data_train = None
        self.data_val = None
        self.seed = seed

    def setup(self, name: str = None):
        # ----- CarRacingDataset is a Dataloader file, takes care of normalization and such-----
        self.data_full = CarRacingDataset(  dataset_path= os.path.join(self.data_dir, name),
                                            pred_horizon=self.T_pred,
                                            obs_horizon=self.T_obs,
                                            action_horizon=self.T_act,
                                            )
        if self.seed:
            self.data_train, self.data_val = random_split(self.data_full, [int(len(self.data_full)*0.8), len(self.data_full) - int(len(self.data_full)*0.8)], generator=torch.Generator().manual_seed(self.seed))
        else:
            self.data_train, self.data_val = random_split(self.data_full, [int(len(self.data_full)*0.8), len(self.data_full) - int(len(self.data_full)*0.8)])
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=4)