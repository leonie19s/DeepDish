import os
import pandas as pd
import random
import torch
from google.cloud import storage
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Any, Dict, Tuple


# Set paths where to download data to
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def download_folder(bucket: storage.Bucket, prefix: str, target_folder_name: str) -> None:
    """
        Helper function that downloads a folder from google cloud storage, acts recursively as the dataset contains
        folders of folders, each of which containing the images
    """

    # Assemble path for overall target folder, create folder if it does not yet exist
    local_dir = os.path.join(DATA_DIR, target_folder_name)
    if os.path.exists(local_dir):
        print(f"Data {prefix} already exists in {local_dir}, skipping download. If you want to download the data again, delete the folder.")
        return
    os.makedirs(local_dir, exist_ok=False)

    # Go through all blobs
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        # Assemble relative path within the blob, join it with local dir to get save path for that blob
        rel_path = os.path.relpath(blob.name, prefix)
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download the blob to the local file
        blob.download_to_filename(local_path)
        print(f'Downloaded {blob.name} to {local_path}')
        

def download_data() -> None:
    """
        Function that downloads the Nutrition5k dataset from the google cloud
    """

    # Init client and get the bucket for the nutrition 5k dataset
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket("nutrition5k_dataset")

    # Don't pull all at once, as the video data is way too large to handle, so only
    # image data and necessary annotations
    download_folder(bucket, 'nutrition5k_dataset/dish_ids', 'dish_ids')
    download_folder(bucket, 'nutrition5k_dataset/metadata', 'metadata')
    download_folder(bucket, 'nutrition5k_dataset/imagery/realsense_overhead', 'images')


def read_annotations(path_to_ann: str) -> pd.DataFrame:
    """
        Function to read in the annotation .csv provided in path_to_ann and return it as pandas DataFrame, one .csv is
        provided per cafeteria where the data was taken from.

        According to the Nutrition5k repository, the annotations are structured like this:
        |
        |    dish_id, total_calories, total_mass, total_fat, total_carb, total_protein, num_ingrs,
        |    (ingr_1_id, ingr_1_name, ingr_1_grams, ingr_1_calories, ingr_1_fat, ingr_1_carb, ingr_1_protein, ...)

        They are thus not regular (i.e. not same number of columns per row), thus we must read it in manually as pd.read_csv
        cannot handle this, we concatenate all individual ingredients to one list and append it to the last column of the DataFrame.
    """

    # Read in every line
    with open(path_to_ann, 'r') as f:
        lines = f.readlines()

    # Parse data line by line, cast to int/float if needed
    data = []
    for line in lines:
        parts = line.strip().split(',')
        dish_id = parts[0]
        total_calories = float(parts[1])
        total_mass = float(parts[2])
        total_fat = float(parts[3])
        total_carb = float(parts[4])
        total_protein = float(parts[5])
        ingredients = parts[6:]

    # Parse individual ingredients one by one, step size 7 according to annotations
    num_ingredients = 0
    for i in range(0, len(ingredients), 7):
        # i = Ingredient id, i+1 = ingredient name, can remain strings
        # Parse each individual ingredient nutrients
        for ii in range(2, 7):
            ingredients[i+ii] = float(ingredients[i+ii])
            num_ingredients += 1

    # Append to datafraem
    data.append({
        'dish_id': dish_id,
        'total_calories': total_calories,
        'total_mass': total_mass,
        'total_fat': total_fat,
        'total_carb': total_carb,
        'total_protein': total_protein,
        'num_ingredients': num_ingredients,
        'ingredients': ingredients,
    })

    # Create a dataframe from it and return
    return pd.DataFrame(data)


def load_data_in_splits(config: Dict[str, Any]) -> Tuple[pd.DataFrame]:
    """
        Function that loads the data, i.e. all image ids and annotations, splits it into train, validation and test set,
        and returns it in that respective order.
        
        We cannot join both cafeterias to to one DataFrame, as it does not have information on total nutrient scores,
        although one could probably infer them from nutrient-wise information, but don't as it is unclear why they did this.

        We cannot use their provided splits as well, as for some reason not every ID in the split is also given as image,
        presumably as the rest is depth data. Thus, we use only those IDs of which we have images, which are 3490 images in total.
    """

    # Read in the data of each cafeteria, and join to one
    df = read_annotations(os.path.join(DATA_DIR, "metadata", "dish_metadata_cafe1.csv"))
    # df2 = read_annotations(os.path.join(DATA_DIR, "metadata", "dish_metadata_cafe2.csv")
    # df = pd.concat([df, df2], ignore_index=True)

    # Create test, validation and train split
    dish_ids = [f for f in os.listdir("data/realsense_overhead")]
    random.shuffle(dish_ids)
    train_split_idx = int(len(dish_ids) * config["train_split_percentage"])
    val_split_idx = int(len(dish_ids) * (config["train_split_percentage"] + config["validation_split_percentage"]))
    train_split_ids = dish_ids[:train_split_idx]
    val_split_ids = dish_ids[train_split_idx:val_split_idx]
    test_split_ids = dish_ids[val_split_idx:]

    # Convert into train and test split and return
    df_train = df[df['dish_id'].isin(train_split_ids)].copy()
    df_val = df[df['dish_id'].isin(val_split_ids)].copy()
    df_test = df[df['dish_id'].isin(test_split_ids)].copy()
    return df_train, df_val, df_test


class FoodDataset(Dataset):
    """
        Dataset class for the Nutrition5k dataset, for use in PyTorch
    """
    def __init__(self, df, img_path, transform=None):
        self.df = df
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # Read in image, parse dish_id and format into the image path template
        row = self.df.iloc[idx]
        img = Image.open(self.img_path.format(row["dish_id"])).convert("RGB")

        # Apply transform if we have one
        if self.transform:
            img = self.transform(img)

        # Create target tensor, for now of total nutritional values
        target = torch.tensor([
            row['total_calories'], row['total_mass'], row['total_fat'], row['total_carb'], row['total_protein']
        ], dtype=torch.float32)

        return img, target


def create_pytorch_loaders(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame]:
    """
        Creates the PyTorch dataloaders from the respective dataframes. Train and val / test transforms are
        already included in this function.
    """

    # Transforms, training with random flipping, test just with norm / resize.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    # Appropriate datasets + loaders, one for each split
    img_path = os.path.join(DATA_DIR, "images")
    train_dataset = FoodDataset(train_df, img_path, transform=train_transform)
    val_dataset = FoodDataset(validation_df, img_path, transform=test_transform)
    test_dataset = FoodDataset(test_df, img_path, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader