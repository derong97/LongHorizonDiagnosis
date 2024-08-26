from toml import load as toml_load
import argparse
from datetime import datetime
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

from embedding_model.config import Config
from embedding_model.dataset import EHR_Train_Dataset, EHR_Val_Dataset, EHR_Test_Dataset
from embedding_model.model import init_model, run_epoch, test_model
from embedding_model.loss import DiscreteFailureTimeNLL, CoxPHLoss
from embedding_model.tune import run_study

def main():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        "--label",
        type=str,
        choices=["autism", "adhd", "ear_infection", "food_allergy"],
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["BC1", "BC2", "BC3", "BC4", "DTNN1", "DTNN2", "DCPH1"],
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tune",
         action=argparse.BooleanOptionalAction,
    )
    
    args = parser.parse_args()

    print("=== Preparing Resources ===")

    config = Config(**toml_load("config.toml"))
    print("Config loaded")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print(f"Training {args.label} {args.model} {config.yob_cutoff} {config.followup_cutoff}")
    # Initialize model
    model = init_model(args.model, config, args.label).to(device)
    print("Model initialized")

    checkpoint_path = args.checkpoint
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Checkpoint loaded")

    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Define loss function
    phenotype = getattr(config.phenotypes, args.label)
    bin_boundaries = phenotype.bin_boundaries
    print(f"Bin boundaries: {bin_boundaries}")
    if args.model.startswith("BC"):
        criterion = BCEWithLogitsLoss()
    elif args.model.startswith("DTNN"):
        criterion = DiscreteFailureTimeNLL(torch.tensor(bin_boundaries, requires_grad=False).to(device))
    elif args.model.startswith("DCPH"):
        criterion = CoxPHLoss()

    # Load train and val dataset/dataloader
    train_dset = EHR_Train_Dataset(config, args.label)
    val_dset = EHR_Val_Dataset(config, args.label)
    print("Loaded train and validation datasets")
    train_dset.describe("train")
    val_dset.describe("val")

    train_loader = DataLoader(train_dset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=config.training.batch_size, shuffle=False)
    print("Loaded train and validation dataloaders")

    # Tune the model
    if args.tune:
        print("=== Model Tuning ===")
        model, params = run_study(config, args.model, args.label, device, train_loader, val_loader, criterion)
        optimizer = optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
        )

    # Train/retrain the model
    train_losses = []
    val_losses = []

    # Early Stopping parameters
    min_val_loss = float("inf")
    early_stopping_counter = 0

    print("=== Training Log ===")
    for epoch in range(config.training.max_epochs):
        train_loss = run_epoch(model, device, train_loader, optimizer, criterion, args.model, training=True)
        val_loss = run_epoch(model, device, train_loader, optimizer, criterion, args.model, training=False)

        print(
            f"Epoch: {epoch} @ {datetime.now()} \n \
        - Train Loss: {train_loss:.3f} - Val Loss: {val_loss:.3f}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # save best checkpoint based on lowest val loss
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            early_stopping_counter = 0

            filename = f"{args.model}{f'_synthetic' if config.experiment.semi_synthetic else ''}_{args.label}_{config.yob_cutoff}_{config.followup_cutoff}_{datetime.now().date()}_epoch={epoch}_val_loss={val_loss:.3f}.pt"
            checkpoint_path = os.path.join("data/checkpoints/logs", filename)
            torch.save(model.state_dict(), checkpoint_path)

        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config.training.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Plot and save results
    plt.plot(np.arange(len(train_losses)), train_losses, label="Train Loss")
    plt.plot(np.arange(len(val_losses)), val_losses, label="Val Loss")
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.legend()

    filename = f"{args.model}{f'_synthetic' if config.experiment.semi_synthetic else ''}_{args.label}_{config.yob_cutoff}_{config.followup_cutoff}_{datetime.now().date()}.png"
    results_path = os.path.join("data/results", filename)
    plt.tight_layout()
    plt.savefig(results_path)
    
    # Evaluate on the test set
    test_dset = EHR_Test_Dataset(config, args.label)
    test_loader = DataLoader(test_dset, batch_size=config.training.batch_size, shuffle=False)
    test_metrics = test_model(model, device, test_loader, bin_boundaries)

    print("=== Results on Test Set ===")
    train_dset.describe("test")
    print(f"Test AUC: {test_metrics['AUC']*100}%")
    print(f"Test AP: {test_metrics['AP']*100}%")
    print(f"Test AUCt: {test_metrics['AUCt']*100}%")
    print(f"Test APt: {test_metrics['APt']*100}%")

if __name__ == "__main__":
    main()