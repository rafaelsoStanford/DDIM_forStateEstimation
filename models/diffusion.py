
import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import numpy as np
from datetime import datetime

# Loading modules
from models.Unet_FiLmLayer import *
from models.simple_Unet import * 
from models.encoder.autoencoder import *
from utils.print_utils import *
from utils.plot_utils import *

class Diffusion(pl.LightningModule):
    def __init__(self
                , noise_steps=1000
                , denoising_steps=1000
                , obs_horizon = 10
                , pred_horizon= 10
                , observation_dim = 2
                , prediction_dim = 2
                , learning_rate = 1e-4
                , model = 'UNet'
                , vision_encoder = None
                , noise_scheduler = 'linear_v2'
                , inpaint_horizon = 10
                 ):
        super().__init__()

        self.save_hyperparameters()
        self.date = datetime.today().strftime('%Y_%m_%d_%H-%M-%S')
# ==================== Init ====================
    # --------------------- Diffusion params ---------------------
        self.noise_steps = self.hparams.noise_steps
        self.denoising_steps = self.hparams.denoising_steps
        self.NoiseScheduler = linear_beta_schedule
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.observation_dim = observation_dim
        self.prediction_dim = prediction_dim
        self.inpaint_horizon = inpaint_horizon

    # --------------------- Model Architecture ---------------------
        if model == 'UNet_Film':
            print("Loading UNet with FiLm conditioning")
            self.model = UNet_Film
        else:
            print("Loading UNet (simple) ")
            self.model = UNet

    # --------------------- Noise Schedule Params---------------------
        if noise_scheduler == 'linear':
            self.NoiseScheduler = linear_beta_schedule
        if noise_scheduler == 'cosine_beta_schedule':
            self.NoiseScheduler = cosine_beta_schedule

        betas =  self.NoiseScheduler(self, noise_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    # --------------------- Model --------------------- 
        self.lr = learning_rate  
        self.loss = nn.MSELoss()
        self.noise_estimator = self.model(
                                    in_channels= 1,
                                    out_channels= 1,
                                    noise_steps= noise_steps,
                                    global_cond_dim= (observation_dim) * obs_horizon, # 512 is the output dim of Resnet18, 2 is the position dim
                                    time_dim = 256 # Embedding dimension for time (t) of the current denoising step
                                )

        ### Define model which will be a simplifed 1D UNet
        if vision_encoder == 'resnet18':
            print("Loading Resnet18")
            self.vision_encoder = VisionEncoder() # Loads pretrained weights of Resnet18 with output dim 512 (also modified layers as Suggested by Song et al.)
        
        else:
            print("Loading lightweight Autoencoder")
            vision = autoencoder.load_from_checkpoint(checkpoint_path="./tb_logs_autoencoder/version_23/checkpoints/epoch=25.ckpt")
            self.vision_encoder = vision.encoder
        self.vision_encoder.device = self.device
        self.vision_encoder.eval() # 128 entries

        # --------------------- Output environment settings ---------------------
        if os.getenv("LOCAL_RANK", '0') == '0':
            print_hyperparameters(
            obs_horizon, pred_horizon, observation_dim, prediction_dim, noise_steps, inpaint_horizon, model, learning_rate, vision_encoder)
            # print("Model Architecture: ", self.noise_estimator)

# ==================== Training ====================
    def training_step(self, batch, batch_idx):
        loss = self.onepass(batch, batch_idx, mode="train")
        self.log("train_loss",loss)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss

# ==================== Testing ====================
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.sample(batch, mode="test")

# ==================== Validation ====================    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.sample(batch, mode="validation")
        loss = self.onepass(batch, batch_idx, mode="validation")
        self.log("val_loss",loss,  sync_dist=True)
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

# ==================== Noising / Denoising Processes ====================

    def onepass(self, batch, batch_idx, mode="train"):
        # ---------------- Preparing Observation / Prediction data ----------------
        x_0 , obs_cond = self.prepare_pred_cond_vectors(batch)
        x_0 = x_0.unsqueeze(1)
        obs_cond = obs_cond.unsqueeze(1)
        B = x_0.shape[0]

        # ---------------- Forward Process ----------------
        t = torch.randint(0, self.noise_steps, (B,), device=self.device).long() # Values from [0, 999]
        noise = torch.randn_like(x_0)
        x_noisy = self.q_forwardProcess(x_0, t, noise) # (B, 1 , pred_horizon, pred_dim)
        x_noisy = self.add_constraints(x_noisy, x_0)

        # ---------------- Estimate noise / Single Backward process ----------------
        # Estimate noise using noise_predictor
        if mode == "train":
            noise_estimated = self.noise_estimator(x_noisy, t, obs_cond)
        else:
            with torch.no_grad():
                noise_estimated = self.noise_estimator(x_noisy, t, obs_cond)

        # ----------------  Loss ----------------
        loss = self.loss(noise, noise_estimated) #MSE Loss
        return loss
    
# ==================== Sampling ====================
    def sample(self, batch, mode):
        # ---------------- Prepare Data ----------------
        x_0 , obs_cond = self.prepare_pred_cond_vectors(batch)
        x_0 = x_0[0,...].unsqueeze(0).unsqueeze(1)
        obs_cond = obs_cond[0,...].unsqueeze(0).unsqueeze(1)
        # Observations ie Past
        position_observation = obs_cond.squeeze()[:, :2].detach().cpu().numpy() 
        actions_observation = obs_cond.squeeze()[:, 2:].detach().cpu().numpy()

        positions_groundtruth = x_0.squeeze()[:, :2].detach().cpu().numpy() #(20 , 2)
        actions_groundtruth = x_0.squeeze()[:, 2:].squeeze().detach().cpu().numpy() #(20 , 3)
    
        # ---------------- Backward Process ----------------
        # Subset of denoising steps
        step_size = 20
        t_subset = torch.arange(0, self.noise_steps, step_size, device=self.device).long() # Values from [0, 999]
        
        # Initial random noise state variable
        self.denoising_steps = len(t_subset) # overwrite the denoising steps size
        x = torch.randn_like(x_0)
        t = t_subset[-1]
        sampling_history = [x.squeeze().detach().cpu().numpy()]
        # ---------------- Backward Process ----------------
        for t_next in reversed(t_subset):
            a_t = self.alphas_cumprod[t]
            a_tnext = self.alphas_cumprod[t_next]
            
            x_1 = a_tnext.sqrt() * (x - (1 - a_t).sqrt() * self.noise_estimator(x, t, obs_cond) ) / a_t.sqrt()
            x_2 = (1 - a_tnext  ).sqrt() * self.noise_estimator(x, t_next, obs_cond)### Currently set ETA to 0 -- DDPM is switched off
            
            x = x_1 + x_2
            x = self.add_constraints(x, x_0)
            sampling_history.append(x.squeeze().detach().cpu().numpy())
            t = t_next # Update t to next element in the subset
        
        # # # ADD LAST DENOISING STEP TO t = 0
        # a_t = self.alphas_cumprod[t]
        # a_tnext = self.alphas_cumprod[0]
        # x_1 = a_tnext.sqrt() * (x - (1 - a_t).sqrt() * self.noise_estimator(x, t, obs_cond) ) / a_t.sqrt()
        # x_2 = (1 - a_tnext  ).sqrt() * self.noise_estimator(x, t, obs_cond)### Currently set ETA to 0 -- DDPM is switched off

        # x = x_1 + x_2
        # x = self.add_constraints(x, x_0)
        # sampling_history.append(x.squeeze().detach().cpu().numpy())
        


        
        plt_toVideo(self,
                sampling_history,
                positions_groundtruth = positions_groundtruth,
                position_observation = position_observation,
                actions_groundtruth = actions_groundtruth,
                actions_observation = actions_observation)

    # q(x_t | x_0)
    def q_forwardProcess(self, x_start, t, noise):
        x_t = torch.sqrt(self.alphas_cumprod[t])[:,None,None,None] * x_start + torch.sqrt(1-self.alphas_cumprod[t])[:,None,None,None] * noise
        return x_t

    @torch.no_grad()
    def p_reverseProcess_loop(self, x_cond, x_0 , x_T = None):
        if x_T is None:
            x_t = torch.rand(1, 1, self.pred_horizon + self.inpaint_horizon, self.prediction_dim, device=self.device)
        else:
            x_t = x_T
        
        for t in reversed(range(0,self.noise_steps)): # t ranges from 999 to 0
            x_t =  self.p_reverseProcess(x_cond,  x_t,  t)

            x_t = self.add_constraints(x_t, x_0)

        return x_t

    @torch.no_grad()
    def p_reverseProcess(self, x_cond, x_t, t):
        if t == 0:
            z = torch.zeros_like(x_t)
        else:
            z = torch.randn_like(x_t)
        est_noise = self.noise_estimator(x_t, torch.tensor([t], device=self.device), x_cond)
        x_t = 1/torch.sqrt(self.alphas[t])* (x_t-(1-self.alphas[t])/torch.sqrt(1-self.alphas_cumprod[t])*est_noise) +  torch.sqrt(self.betas[t])*z
        return x_t

    def add_constraints(self, x_t , x_0):
        # Adding constraints by inpainting before denoising. 
        # Add all constaints here
        x_t[:, : , :self.inpaint_horizon, :] = x_0[:, : , :self.inpaint_horizon, :].clone() # inpaint datapoints 
        # x_t[:, :, :, 2] = torch.clip(x_t[:, :, :, 2].clone(), min=-1.0, max=1.0) # Enforce action limits (steering angle)
        # x_t[:, :, :, 3:] = torch.clip(x_t[:, :, :, 3:].clone(), min=-1.0, max=1.0)   # Enforce action limits (acceleration and brake)
        return x_t

    # ==================== Helper functions ====================
    def prepare_pred_cond_vectors(self, batch):
        # Security check for corrupted data
        assert(not torch.isnan(batch['position'][:,: self.obs_horizon ,:]).any())
        assert(not torch.isnan(batch['action'][:,self.obs_horizon : ,:]).any())

        # ---------------- Preparing Observation data ----------------
        normalized_img = batch['image'][:,:self.obs_horizon ,:] 
        normalized_pos = batch['position'][:,:self.obs_horizon ,:]
        normalized_act = batch['action'][:,:self.obs_horizon ,:]
        normalized_vel = batch['velocity'][:,:self.obs_horizon ,:]

        # ---------------- Encoding Image data ----------------
        encoded_img = self.vision_encoder(normalized_img.flatten(end_dim=1)) # (B, 128)
        image_features = encoded_img.reshape(*normalized_img.shape[:2],-1) # (B, t_0:t_obs , 128)

        # ---------------- Conditional vector ----------------
        # Concatenate position and action data and image features
        obs_cond = torch.cat([normalized_pos, normalized_act,normalized_vel, image_features], dim=-1) # (B, t_0:t_obs, 512 + 3 + 2)

        # ---------------- Preparing Prediction data (acts as ground truth) ----------------
        x_0_pos = batch['position'][:,self.obs_horizon: ,:] # (B, t_obs:t_pred , 2)
        x_0_act = batch['action'][:, self.obs_horizon: ,:] # (B, t_obs:t_pred, 3)
        x_0 = torch.cat([x_0_pos, x_0_act], dim=-1) # (B, t_obs:t_pred, 5)

        # Adding past obervation as inpainting condition
        x_0 = torch.cat((obs_cond[:, -self.inpaint_horizon:, :5], x_0) , dim=1) # Concat in time dim

        # ---------------- Assert cond dimensions compatible with model (important when preloading / changing conditioning data) ----------------
        assert(obs_cond.shape[-1]*self.obs_horizon == self.noise_estimator.down1.cond_encoder[2].state_dict()['weight'].shape[1]) # Check if cond dim is correct
        return x_0 , obs_cond

# ==================== Schedulers ====================
def linear_beta_schedule(self, steps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    beta = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32, device=self.device)
    return beta


def cosine_beta_schedule(self, timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    betas =  torch.tensor(betas_clipped, dtype=dtype)
    return betas
