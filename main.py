import torch
from data import download_data, load_data_in_splits, create_pytorch_loaders
from deepdish import DeepDish
from training import train, eval_model


# Run config, change settings here
CONFIG = {
    "run_name": "UnfreezeLastTwoLayers250Epochs",
    "backbone": "dinov2",
    "checkpoint": None,
    "device": "cuda",
    "epochs": 250,
    "learning_rate": 4e-4,
    "batch_size": 32,
    "head_hidden_size": 384,
    "head_dropout_p": 0.1,
    "unfreeze_backbone_block_after_n": 10,
    "train_split_percentage": 0.85,
    "validation_split_percentage": 0.05,
}


def main(config):
    # Make sure the data exists, download if it does not
    download_data()

    # Load data and create splits from it, also the respective loaders
    df_train, df_val, df_test = load_data_in_splits(config)
    train_loader, val_loader, test_loader = create_pytorch_loaders(df_train, df_val, df_test, config)

    # Create the model, load in checkpoint if provided, cast to device
    model = DeepDish(5, config)
    if config["checkpoint"]:
        print("Loading in model...")
        model.load_state_dict(torch.load(config["checkpoint"]))
    model.to(config["device"])
    model.set_requires_grad()

    # Main training loop and subsequent test set evaluation
    train(model, config, train_loader, val_loader)
    test_loss, test_indiv_losses = eval_model(model, test_loader)

    print("\n==== TEST SET STATISTICS ====\n")
    print(f"Mean Squared Error: {test_loss}")
    for i, nutr in enumerate(["Calories", "Mass", "Fat", "Carb", "Protein"]):
        print(f"Mean Absolute Error for {nutr}: {test_indiv_losses[i]}")


if __name__ == "__main__":
    main(CONFIG)