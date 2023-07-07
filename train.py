import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint

from models.diffusion import *
from utils.load_data import *
from utils.print_utils import *

# =========== parser function ===========
def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--n_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision (AMP)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    parser.add_argument('--obs_horizon', type=int, default=40, help='Observation horizon')
    parser.add_argument('--pred_horizon', type=int, default=40, help='Prediction horizon')
    parser.add_argument('--action_horizon', type=int, default=1, help='Action horizon')
    parser.add_argument('--inpaint_horizon', type=int, default=5, help='Inpaining horizon, which denotes the amount of steps of our observations to use for inpainting')
    parser.add_argument('--noise_steps', type=int, default=1000, help='Denoising steps')
    
    parser.add_argument('--cond_dim', type=int, default=128+2+3+2, help='Dimension of diffusion input state')
    parser.add_argument('--output_dim', type=int, default=5, help='Dimension of diffusion output state')
    parser.add_argument('--model', type=str, default='UNet_Film', help='String for choosing model architecture')
    parser.add_argument('--noise_scheduler', type=str, default='linear', help='String for choosing noise scheduler')

    parser.add_argument('--dataset_dir', type=str, default='./data', help='Path to dataset directory')
    parser.add_argument('--dataset', type=str, default='Sinusoidal_dataset_5_episodes.zarr.zip', help='zarr.zip dataset filename')
    
    return parser.parse_args()


############################
#========== MAIN ===========
############################
def main(args):

    # ===========Parameters===========
    # training parameters
    n_epochs = args.n_epochs
    AMP = args.amp
    batch_size = args.batch_size
    lr = args.lr

    # diffusion parameters
    noise_steps = args.noise_steps
    obs_horizon = args.obs_horizon
    pred_horizon = args.pred_horizon
    inpaint_horizon = args.inpaint_horizon
    action_horizon = args.action_horizon
    cond_dim = args.cond_dim
    output_dim = args.output_dim


    # Dataset dir and filename
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset

    # model architecture
    model = args.model
    noise_scheduler = args.noise_scheduler
    
    # =========== Loading Data ===========
    # Load Dataset using Pytorch Lightning DataModule
    dataset = CarRacingDataModule(batch_size, dataset_dir , obs_horizon, pred_horizon ,action_horizon)
    
    dataset.setup(name=dataset_name)
    train_dataloader = dataset.train_dataloader()
    valid_dataloader = dataset.val_dataloader()

    # # ===========model===========
    diffusion = Diffusion(
                    noise_steps= noise_steps,
                    obs_horizon=obs_horizon,
                    pred_horizon= pred_horizon,
                    observation_dim=cond_dim,
                    prediction_dim= output_dim,
                    model=model,
                    learning_rate=lr,
                    inpaint_horizon=inpaint_horizon,
                    noise_scheduler=noise_scheduler,
    )

    # ===========trainer===========
    # -----PL configs-----
    tensorboard = pl_loggers.TensorBoardLogger(save_dir="tb_logs/",name='',flush_secs=1)

    early_stop_callback = EarlyStopping(monitor='lr', stopping_threshold=2e-6, patience=n_epochs)   
    checkpoint_callback = ModelCheckpoint(filename="{epoch}",     # Checkpoint filename format
                                          save_top_k=-1,          # Save all checkpoints
                                          every_n_epochs=1,               # Save every epoch
                                          save_on_train_epoch_end=True,
                                          verbose=True)
    # -----train model-----
    trainer = pl.Trainer(accelerator='gpu', devices=[0,1], precision=("16-mixed" if AMP else 32), max_epochs=n_epochs, 
                         callbacks=[early_stop_callback, checkpoint_callback],
                         logger=tensorboard, profiler="simple", val_check_interval=0.25, 
                         accumulate_grad_batches=1, gradient_clip_val=0.5) #, strategy='ddp_find_unused_parameters_true') 

    # -----print model info------
    if os.getenv("LOCAL_RANK", '0') == '0':
        print_dataset_info(args,dataset_dir, dataset_name, train_dataloader, tensorboard)

    trainer.validate(model= diffusion, dataloaders=valid_dataloader)
    trainer.fit(model=diffusion, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)





