
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint

import zarr
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import transforms as T






class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str):
        self.dataset_path = dataset_path
        dataset_root = zarr.open(self.dataset_path, 'r')
        image_data = dataset_root['data']['img'][:]
        self.image_data = image_data # [0,1] images are already normalized



    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        image = self.image_data[idx]
        return image

# data module
class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.data_train = None
        self.data_val = None

    def setup(self, name: str = None):
        # ----- CarRacingDataset is a Dataloader file, takes care of normalization and such-----
        self.data_full = ImagesDataset(self.data_dir)
        self.data_train, self.data_val = random_split(self.data_full, [int(len(self.data_full)*0.8), len(self.data_full) - int(len(self.data_full)*0.8)])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=4)


dataset = DataModule('/home/rafael/git_repos/diffusion_bare/data/oldDatasets/multipleDrivingBehaviours_testing_5eps_normalized.zarr.zip')
dataset.setup()
train_dataloader = dataset.train_dataloader()
val_dataloader = dataset.val_dataloader()


# model

model = AE()
trainer = pl.Trainer()
trainer.fit(model)