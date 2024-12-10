import yaml
from pathlib import Path
import torch
from src.data.download import download_plantdoc_dataset
from src.data.preprocess import PlantDocPreprocessor
from src.data.dataloader import get_dataloaders
from src.models.trainer import PlantDocTrainer
from src.eval.evaluate import PlantDocEvaluator


def load_config():
    """設定ファイルの読み込み"""
    with open("configs/training_config.yaml", "r") as f:
        return yaml.safe_load(f)


def setup_directories():
    """必要なディレクトリの作成"""
    directories = [
        "data/raw",
        "outputs/checkpoints",
        "outputs/logs",
        "outputs/evaluation"
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def main():
    # 設定の読み込み
    config = load_config()

    # ディレクトリの設定
    setup_directories()

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データセットのダウンロード
    print("Downloading dataset...")
    download_plantdoc_dataset()

    # データの前処理
    print("Preprocessing data...")
    preprocessor = PlantDocPreprocessor(config)
    train_df, val_df, test_df = preprocessor.process_dataset()

    # データローダーの作成
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, config, preprocessor
    )

    # トレーナーの初期化
    print("Initializing trainer...")
    trainer = PlantDocTrainer(config)

    # モデルの学習
    print("Starting training...")
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config["training"]["num_epochs"]
    )

    # モデルの評価
    print("Evaluating model...")
    evaluator = PlantDocEvaluator(
        trainer.model,
        config,
        trainer.processor,
        device
    )
    evaluator.evaluate(test_loader, list(preprocessor.classes.keys()))

    print("Training and evaluation completed!")


if __name__ == "__main__":
    main()
