import json
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_SAVE_PATH = os.path.join(BASE_DIR, "logs")


def eval_model(model, loader):
    with torch.no_grad():

        model.eval()
        mse_loss = nn.MSELoss()

        # Collecting statistics for each epoch
        val_loss = 0.
        val_indiv_losses = torch.tensor([0., 0., 0., 0., 0.], dtype=torch.float32)
        its = 0

        for batch, (image, target) in enumerate(loader):

            its += 1

            # Get model output
            image = image.cuda()
            target = target.cuda()
            output = model(image)

            # Compute the loss
            loss = mse_loss(output, target)
            val_loss += loss.clone().detach().item()

            # Compute nutrition-wise differences and add to collector
            val_indiv_losses += torch.mean(torch.abs(output.clone().detach().cpu() - target.clone().detach().cpu()), dim=0)

        val_loss /= its
        val_indiv_losses /= its

        return val_loss, val_indiv_losses


def train(
    model,
    config,
    train_loader,
    val_loader
):
    # Create folder where data is saved if it does not exist
    os.makedirs(TRAINING_SAVE_PATH, exist_ok=True)

    # Loss: Mean Squared Error between model output and target
    criterion = nn.MSELoss()

    # Use Adam as optimizer, for the parameters that are to be trained
    optimizer = Adam([
        {'params': (param for name, param in model.named_parameters() if param.requires_grad), 'lr': config["learning_rate"]},
    ])

    # Create summary writer for tensorboard
    log_dir = os.path.join(TRAINING_SAVE_PATH, config["run_name"])
    writer = SummaryWriter(log_dir)
    config_str = json.dumps(config, indent=4)
    writer.add_text('hyperparameters/config', config_str)
    epoch_losses = []

    # Save config as json
    with open(os.path.join(log_dir, "config.json"), "w") as json_file:
        json.dump(config, json_file, indent=4)

    # Main training loop
    for epoch in tqdm(range(config["epochs"])):

        # Collecting statistics for each epoch
        epoch_loss = 0.
        epoch_indiv_losses = torch.tensor([0., 0., 0., 0., 0.], dtype=torch.float32)
        its = 0

        for batch, (image, target) in enumerate(train_loader):

            model.train()
            its += 1

            # Cast to proper device, get model output
            image = image.to(config["device"])
            target = target.to(config["device"])
            optimizer.zero_grad()
            output = model(image)

            # Compute the loss
            loss = criterion(output, target)
            epoch_loss += loss.clone().detach().item()

            # Compute nutrition-wise differences and add to collector
            epoch_indiv_losses += torch.mean(torch.abs(output.clone().detach().cpu() - target.clone().detach().cpu()), dim=0)

            # Backward the loss, take a step with the optimizer to upgrade the model weights
            loss.backward()
            optimizer.step()

        # Calculate epoch-wise statistics and write to tensorboard
        epoch_loss /= its
        epoch_indiv_losses /= its
        writer.add_scalar('TRAIN/loss', epoch_loss, epoch)

        # Validation, also write to tensorboard, calculate nutrient-wise losses for both splits
        val_loss, val_indiv_losses = eval_model(model, val_loader)
        writer.add_scalar('VAL/loss', val_loss, epoch)
        for i, nutr in enumerate(["calories", "mass", "fat", "carb", "protein"]):
            writer.add_scalar('TRAIN/' + nutr, epoch_indiv_losses[i], epoch)
            writer.add_scalar('VAL/' + nutr, val_indiv_losses[i], epoch)

        print(f" Train Loss: {epoch_loss}, Val Loss: {val_loss}")

    # Save model
    torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))