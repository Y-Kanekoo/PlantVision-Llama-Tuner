import os
import json
import argparse
import logging
from pathlib import Path
import random
import shutil
import zipfile
from typing import Dict, List, Tuple, Optional
import requests
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import re

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlantDocPreprocessor:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.data_dir = Path("data")
        self.cache_dir = Path("cache")
        self.dataset_url = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"

        # データディレクトリの作成
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        # 変換の設定
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.05
            ),
            transforms.RandomRotation(20, fill=255),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_config(self, config_path: str) -> dict:
        """設定ファイルを読み込む"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def download_dataset(self):
        """データセットをダウンロードして展開"""
        zip_path = self.data_dir / "plantdoc.zip"

        if not zip_path.exists():
            logger.info("データセットをダウンロードしています...")
            response = requests.get(self.dataset_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)

        logger.info("データセットを展開しています...")
        extract_dir = self.data_dir / "temp_extract"
        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # ファイル名を正規化して展開
            for file_info in zip_ref.filelist:
                # 不正な文字を除去
                clean_name = re.sub(r'\?.*\.jpg$', '.jpg', file_info.filename)
                clean_name = clean_name.replace('?', '')

                try:
                    # ファイル名が変更された場合
                    if clean_name != file_info.filename:
                        data = zip_ref.read(file_info.filename)
                        clean_path = extract_dir / clean_name
                        clean_path.parent.mkdir(parents=True, exist_ok=True)
                        clean_path.write_bytes(data)
                    else:
                        # 通常の展開
                        zip_ref.extract(file_info, extract_dir)
                except Exception as e:
                    logger.warning(
                        f"ファイルの展開中にエラー: {file_info.filename}, エラー: {str(e)}")
                    continue

        # 展開したファイルを正しい場所に移動
        final_dir = self.data_dir / "PlantDoc-Dataset-master"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.move(str(extract_dir / "PlantDoc-Dataset-master"),
                    str(final_dir))

        # 一時ディレクトリとZIPファイルを削除
        shutil.rmtree(extract_dir)
        zip_path.unlink()
        logger.info("データセットのダウンロードと展開が完了しました")

    def prepare_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """データセットの準備とトレーニング/検証/テストデータの分割"""
        dataset_path = self.data_dir / "PlantDoc-Dataset-master"
        train_path = dataset_path / "train"
        test_path = dataset_path / "test"
        all_data = []

        # クラスごとのカウンターを初期化
        class_counts = {
            label: 0 for label in self.config["data"]["label_mapping"]}

        # トレーニングデータの読み込み
        for label_dir in tqdm(train_path.iterdir(), desc="トレーニングデータの読み込み"):
            if not label_dir.is_dir() or label_dir.name.startswith('.'):
                continue

            label = self._normalize_label(label_dir.name)
            if label not in self.config["data"]["label_mapping"]:
                logger.warning(f"未知のラベル: {label}")
                continue

            for img_path in label_dir.glob("*.jpg"):
                all_data.append({
                    "image_path": str(img_path),
                    "label": label,
                    "label_en": self.config["data"]["label_mapping"][label]["en"],
                    "label_ja": self.config["data"]["label_mapping"][label]["ja"],
                    "split": "train"
                })
                class_counts[label] += 1

        # テストデータの読み込み
        for label_dir in tqdm(test_path.iterdir(), desc="テストデータの読み込み"):
            if not label_dir.is_dir() or label_dir.name.startswith('.'):
                continue

            label = self._normalize_label(label_dir.name)
            if label not in self.config["data"]["label_mapping"]:
                logger.warning(f"未知のラベル: {label}")
                continue

            for img_path in label_dir.glob("*.jpg"):
                all_data.append({
                    "image_path": str(img_path),
                    "label": label,
                    "label_en": self.config["data"]["label_mapping"][label]["en"],
                    "label_ja": self.config["data"]["label_mapping"][label]["ja"],
                    "split": "test"
                })
                class_counts[label] += 1

        # クラス分布のログ出力
        logger.info("クラスごとの画像数:")
        for label, count in class_counts.items():
            logger.info(f"{label}: {count}枚")

        # データのシャッフルと分割
        random.seed(self.config["data"]["random_seed"])
        train_data = [item for item in all_data if item["split"] == "train"]
        test_data = [item for item in all_data if item["split"] == "test"]

        # 検証データの作成（トレーニングデータの15%）
        random.shuffle(train_data)
        val_size = int(len(train_data) * 0.15)
        val_data = train_data[:val_size]
        train_data = train_data[val_size:]

        logger.info(f"トレーニングデータ: {len(train_data)}件")
        logger.info(f"検証データ: {len(val_data)}件")
        logger.info(f"テストデータ: {len(test_data)}件")

        return train_data, val_data, test_data

    def _normalize_label(self, label: str) -> str:
        """ラベル名を正規化する"""
        label = label.lower().strip()
        label = label.replace("leaf", "").strip()
        label = label.replace("_", "").strip()

        # マッピング
        label_map = {
            "tomato two spotted spider mites": "spider_mites",
            "tomato mold": "leaf_mold",
            "tomato": "healthy",
            "tomato yellow virus": "yellow_leaf_curl_virus",
            "tomato mosaic virus": "mosaic_virus",
            "tomato late blight": "late_blight",
            "tomato bacterial spot": "bacterial_spot",
            "tomato septoria": "septoria_leaf_spot",
            "tomato early blight": "early_blight"
        }

        return label_map.get(label, label)

    def process_chunk(self, data: List[Dict], transform, desc: str) -> List[Dict]:
        """データのチャンク処理"""
        processed_data = []
        for item in tqdm(data, desc=desc):
            try:
                image = Image.open(item["image_path"]).convert("RGB")
                if transform:
                    image = transform(image)

                processed_data.append({
                    "image": image,
                    "label": item["label"],
                    "label_en": item["label_en"],
                    "label_ja": item["label_ja"]
                })
            except Exception as e:
                logger.warning(
                    f"画像の処理中にエラーが発生: {item['image_path']}, エラー: {str(e)}")
                continue

        return processed_data


class PlantDocDataset(Dataset):
    def __init__(self, data: List[Dict], config: dict, transform=None, split="train"):
        self.data = data
        self.config = config
        self.transform = transform
        self.split = split

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        # 両言語のプロンプトを準備
        prompts = {}
        for lang in self.config["data"]["languages"]:
            prompt_config = self.config["data"]["prompts"][lang]
            condition = self.config["data"]["label_mapping"][item["label"]][lang]

            prompt = f"{prompt_config['system']}\n\n"
            prompt += prompt_config["instruction"].format(condition=condition)
            prompts[lang] = prompt

        return {
            "image": item["image"],
            "label": item["label"],
            "prompts": prompts,
            "response_formats": {
                lang: self.config["data"]["prompts"][lang]["response_format"]
                for lang in self.config["data"]["languages"]
            }
        }


def main():
    parser = argparse.ArgumentParser(description="PlantDocデータセットの前処理")
    parser.add_argument("--download", action="store_true",
                        help="データセットをダウンロード")
    parser.add_argument("--preprocess", action="store_true", help="データの前処理を実行")
    args = parser.parse_args()

    preprocessor = PlantDocPreprocessor()

    if args.download:
        preprocessor.download_dataset()

    if args.preprocess:
        # データの準備と分割
        train_data, val_data, test_data = preprocessor.prepare_data()

        # データの処理とキャッシュ
        logger.info("データの前処理を実行中...")

        processed_train = preprocessor.process_chunk(
            train_data,
            preprocessor.train_transform,
            "トレーニングデータの処理"
        )
        processed_val = preprocessor.process_chunk(
            val_data,
            preprocessor.eval_transform,
            "検証データの処理"
        )
        processed_test = preprocessor.process_chunk(
            test_data,
            preprocessor.eval_transform,
            "テストデータの処理"
        )

        # 処理済みデータの保存
        torch.save({
            "train": processed_train,
            "val": processed_val,
            "test": processed_test,
            "config": preprocessor.config
        }, preprocessor.cache_dir / "processed_data.pt")

        logger.info("データの前処理が完了しました")


if __name__ == "__main__":
    main()
