import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models


class Autoencoder(nn.Module):
    def __init__(self, channels=3, latent_dim = 128):
        super().__init__()        
        # N, 3, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, 2 , stride=2, padding=1), # -> N, 16, 48, 48
            nn.ReLU(),
            nn.Conv2d(16, 32, 2, stride=2, padding=0), # -> N, 32, 24, 24 
            nn.ReLU(),
            nn.Conv2d(32, 64, 2, stride=2, padding=0), # -> N, 64, 12, 12
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*12*12, latent_dim)
        )
        
        # N , 64, 1, 1
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64*12*12,), # -> N, 64*12*12
            nn.Unflatten(1, (64, 12, 12)), # -> N, 64, 12, 12
            nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0),  # N, 32, 24, 24 (N,32,23,23 without output_padding)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0),  # N, 16, 48, 48 (N,16,47,47)
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, 2, stride=2, padding=0),   # N, 3, 96, 96 (N,3,95,95)
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

class autoencoder(pl.LightningModule):
    def __init__( self, learning_rate=1e-3):
        super().__init__()

    # ==================== Init ====================

    # --------------------- Model --------------------- 
        self.lr = learning_rate  
        self.loss = nn.MSELoss()
        self.model = Autoencoder()
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
    
    # ==================== Training ====================

    def onepass(self, batch, batch_idx, mode="train"):
        recon = self.model(batch)
        loss = self.loss(recon, batch)
        return loss
    

    def training_step(self, batch, batch_idx):
        loss = self.onepass(batch, batch_idx, mode="train")
        self.log("train_loss",loss)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss
    
# ==================== Validation ====================    
    def validation_step(self, batch, batch_idx):
        loss = self.onepass(batch, batch_idx, mode="validation")
        self.log("val_loss",loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5) # patience in the unit of epoch
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }


