import torch
import pytorch_lightning as pl
from datetime import datetime

def print_hyperparameters(obs_horizon, pred_horizon, observation_dim, prediction_dim, noise_steps, inpaint_horizon, model, learning_rate, vision_encoder):
    print("*" * 20 + " OVERVIEW " +  "*" * 20)
    print()
    print("======== Hyperparameters =======")
    print("Date:", datetime.today().strftime('%Y-%m-%d-%H'))
    print("Observation Horizon:", obs_horizon)
    print("Prediction Horizon:", pred_horizon)
    print("Observation Dimension:", observation_dim)
    print("Prediction Dimension:", prediction_dim)
    print("Noise Steps:", noise_steps)
    print("Inpaint Horizon:", inpaint_horizon)

    print("======== Model Architecture =======")
    if model == 'UNet_Film':
        print("Model: UNet with FiLm conditioning")
    else:
        print("Model: UNet (simple)")
    print(" Learning Rate:", learning_rate)

    print("======== Model =======")
    print("Noise Estimator Model:")
    print("- In Channels:", 1)
    print("- Out Channels:", 1)
    print("- Noise Steps:", noise_steps)
    print("- Global Conditioning Dimension:", (observation_dim) * obs_horizon)
    print("- Time Dimension:", 256)

    print("Vision Encoder Model:")
    if vision_encoder == 'resnet18':
        print("Vision Encoder: Resnet18")
    else:
        print("Vision Encoder: Lightweight Autoencoder")
    print()
    print("*" * 40)


# # =========== DATA OUTPUT FUNCTION ===========

def print_dataset_info(args, dataset_dir, dataset_name, dataloader, logger):
    # Dataset information
    print("========= Dataset Information =========")
    print("Dataset Directory:", dataset_dir)
    print("Dataset Name:", dataset_name)
    # Load the dataset
    print("Dataset Length:", len(dataloader))

    # TensorBoard information
    print("\n========= TensorBoard Information =========")
    print("TensorBoard Log Directory:", logger.log_dir)

    # Lightning Trainer information
    print("\n========= Lightning Trainer Information =========")
    print("Max Epochs:", args.n_epochs)
    print("Batch Size:", args.batch_size)
    print("Learning Rate:", args.lr)
    print("Accelerator:", "GPU" if torch.cuda.is_available() else "CPU")

    # Device information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU Name:", torch.cuda.get_device_name(device))

    # PyTorch Lightning information
    pl.seed_everything(42)  # Set random seed for reproducibility
    print("\n========= PyTorch Lightning Information =========")
    print("PyTorch Lightning Version:", pl.__version__)
