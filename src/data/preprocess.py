import os
from pathlib import Path
import cv2
import albumentations as A
from PIL import Image
import json
from sklearn.model_selection import train_test_split
import pandas as pd


class PlantDocPreprocessor:
    def __init__(self, config):
        self.config = config
        self.transform = self._get_transforms()
        self.classes = self._get_classes()

    def _get_classes(self):
        """クラスの取得（13種の植物 x 17種の病害）"""
        raw_data_dir = Path("data/raw")
        classes = [d.name for d in raw_data_dir.iterdir() if d.is_dir()]
        return {cls: idx for idx, cls in enumerate(sorted(classes))}

    def _get_transforms(self):
        """データ拡張の設定"""
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def process_dataset(self):
        """データセットの前処理とトレーニング用フォーマットへの変換"""
        raw_data_dir = Path("data/raw")
        processed_data = []

        for class_dir in raw_data_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                class_id = self.classes[class_name]

                for img_path in class_dir.glob("*.jpg"):
                    processed_data.append({
                        "image_path": str(img_path),
                        "class_name": class_name,
                        "class_id": class_id
                    })

        df = pd.DataFrame(processed_data)
        return self._split_dataset(df)

    def _split_dataset(self, df):
        """データセットの分割"""
        train_df, temp_df = train_test_split(
            df,
            test_size=self.config["data"]["train_test_split"],
            stratify=df["class_id"],
            random_state=self.config["data"]["seed"]
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df["class_id"],
            random_state=self.config["data"]["seed"]
        )

        return train_df, val_df, test_df
