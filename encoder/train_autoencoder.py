
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

import torch.nn as nn
import torch.nn.functional as F

from autoencoder import *

EVAL = False


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str):
        self.dataset_path = dataset_path
        dataset_root = zarr.open(self.dataset_path, 'r')
        image_data = dataset_root['data']['img'][:]
        image_data = np.moveaxis(image_data, -1, 1) 
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
        self.save_hyperparameters()

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


dataset = DataModule('/home/rafael/git_repos/diffusion_bare/data/oldDatasets/multipleDrivingBehaviours_testing_5eps_normalized.zarr.zip', batch_size=128)
dataset.setup()
train_dataloader = dataset.train_dataloader()
val_dataloader = dataset.val_dataloader()
n_epochs = 50

# ===========trainer===========
# -----PL configs-----
tensorboard = pl_loggers.TensorBoardLogger(save_dir="tb_logs_autoencoder/",name='',flush_secs=1)

model = autoencoder()

early_stop_callback = EarlyStopping(monitor='lr', stopping_threshold=2e-6, patience=n_epochs)   
checkpoint_callback = ModelCheckpoint(filename="{epoch}",     # Checkpoint filename format
                                        save_top_k=-1,          # Save all checkpoints
                                        every_n_epochs=1,              # Save every epoch
                                        save_on_train_epoch_end=True,
                                        verbose=True)
# -----train model-----
trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=n_epochs, 
                        callbacks=[early_stop_callback, checkpoint_callback],
                        logger=tensorboard, profiler="simple", val_check_interval=0.5,
                        accumulate_grad_batches=1, gradient_clip_val=0.5 ) 

trainer.validate(model= model, dataloaders=val_dataloader)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


# ===========test model===========
if EVAL:
    # choose your trained nn.Module
    encoder = model.encoder
    decoder = model.decoder
    encoder.eval()
    decoder.eval()

    # Get batch random images from the validation set
    batch = next(iter(val_dataloader))
    with torch.no_grad():
        embeddings = encoder(batch)
        print("⚡" * 20, "\nPredictions (batch image embeddings):\n", embeddings.shape, "\n", "⚡" * 20)
        reconstructions = decoder(embeddings)

        # Reshape the images to (batch_size, height, width, channels)
        images = reconstructions.permute(0, 2, 3, 1).detach().numpy()

        # Create a figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(16, 3))

        # Set axis labels and title
        ax.set_title("Reconstructions")

        w = 94
        h = 94

        fig = plt.figure(figsize=(20, 20))
        columns = 8
        rows = 2
        for i in range(1, columns*rows +1):
            img = images[i, :]
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        # Show the plot
        plt.show()