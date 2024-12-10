import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np


class PlantDocDataset(Dataset):
    def __init__(self, df, preprocessor, is_train=True):
        self.df = df
        self.preprocessor = preprocessor
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.is_train:
            augmented = self.preprocessor.transform(image=image)
            image = augmented["image"]

        return {
            "image": torch.tensor(image, dtype=torch.float32).permute(2, 0, 1),
            "label": torch.tensor(row["class_id"], dtype=torch.long)
        }


def get_dataloaders(train_df, val_df, test_df, config, preprocessor):
    train_dataset = PlantDocDataset(train_df, preprocessor, is_train=True)
    val_dataset = PlantDocDataset(val_df, preprocessor, is_train=False)
    test_dataset = PlantDocDataset(test_df, preprocessor, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, test_loader
