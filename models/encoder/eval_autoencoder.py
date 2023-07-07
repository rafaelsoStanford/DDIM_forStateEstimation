
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split

import zarr
import numpy as np
import torch
import matplotlib.pyplot as plt
from autoencoder import *

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

model = autoencoder.load_from_checkpoint(
    checkpoint_path="./tb_logs_autoencoder/version_23/checkpoints/epoch=25.ckpt",
)

model.eval()
encoder = model.encoder
decoder = model.decoder
encoder.eval()
decoder.eval()


dataset = DataModule('/home/rafael/git_repos/diffusion_bare/data/oldDatasets/multipleDrivingBehaviours_testing_5eps_normalized.zarr.zip', batch_size=128)
dataset.setup()
train_dataloader = dataset.train_dataloader()
val_dataloader = dataset.val_dataloader()



# Get batch random images from the validation set
batch = next(iter(val_dataloader))
batch = batch.to(model.device)

with torch.no_grad():
    embeddings = encoder(batch)
    print("⚡" * 20, "\nPredictions (batch image embeddings):\n", embeddings.shape, "\n", "⚡" * 20)
    reconstructions = decoder(embeddings)

    # Reshape the images to (batch_size, height, width, channels)
    images_orig = batch.permute(0, 2, 3, 1).cpu().detach().numpy()
    images = reconstructions.permute(0, 2, 3, 1).cpu().detach().numpy()

    # Create a figure and axis
    #fig, ax = plt.subplots(1, 1, figsize=(16, 3))

    # Set axis labels and title
    
    # Display first image
    fig, axs = plt.subplots(1, 2)

    fig = plt.figure(figsize=(20, 20))
    columns = 8
    rows = 2
    for i in range(1, columns*1 +1):
        img = images_orig[i, :]
        fig.add_subplot(1, columns, i)
        plt.imshow(img)

    for i in range(1, columns +1):
        img = images[i, :]
        fig.add_subplot(2, columns, i)
        plt.imshow(img)
    # Show the plot
    plt.show()