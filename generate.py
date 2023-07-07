
import pytorch_lightning as pl
from models.diffusion import *
from utils.load_data import *

import yaml

def fetch_hyperparams_from_yaml(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams


############################
#========== MAIN ===========
############################

def main():
    # Some params, dont know where to put them
    AMP = True
    n_epochs = 1
    batch_size = 1
    denoising_steps = 1000

    # =========== Load Model ===========
    path_hyperparams = './tb_logs/version_513/hparams.yaml'
    path_checkpoint = './tb_logs/version_513/checkpoints/epoch=45.ckpt'

    model_params = fetch_hyperparams_from_yaml(path_hyperparams)
    model = Diffusion.load_from_checkpoint(
        path_checkpoint,
        hparams_file=path_hyperparams,
        denoising_steps = denoising_steps,
    )
    model.eval() 

    # ===========Parameters===========
    obs_horizon = model_params['obs_horizon']
    pred_horizon = model_params['pred_horizon']

    # =========== Dataloader ===========
    # Dataset dir and filename
    dataset_dir = './data'
    dataset = CarRacingDataModule( batch_size, dataset_dir, obs_horizon, pred_horizon ,seed=42)
    dataset.setup( name='Sinusoidal_dataset_5_episodes.zarr.zip' )
    test_dataloaders = dataset.val_dataloader()

    # =========== Pytorch Lightning Trainer  ===========

    trainer = pl.Trainer(accelerator='gpu', devices=[0], precision=("16-mixed" if AMP else 32), max_epochs=n_epochs)
    trainer.test(model, dataloaders=test_dataloaders)

if __name__ == "__main__":
    main()
    