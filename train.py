import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from unsloth import FastLlamaModel
from accelerate import Accelerator
from peft import LoraConfig
import wandb
from tqdm import tqdm
from data_preprocessing import PlantDocDataset
import time
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, MllamaForConditionalGeneration, AutoProcessor

# 環境変数の読み込み
load_dotenv()

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlantVisionTrainer:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # wandbの設定
        if self.config["wandb"]["project"]:
            wandb.init(
                project=self.config["wandb"]["project"],
                config=self.config
            )

    def _load_config(self, config_path: str) -> dict:
        """設定ファイルを読み込む"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_datasets(self) -> tuple:
        """データセットの読み込み"""
        logger.info("データセットを読み込んでいます...")
        cache_path = Path("cache") / "processed_data.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                "前処理済みデータが見つかりません。data_preprocessing.pyを実行してください。")

        cached_data = torch.load(cache_path)
        train_dataset = PlantDocDataset(
            cached_data["train"], self.config, split="train")
        val_dataset = PlantDocDataset(
            cached_data["val"], self.config, split="val")
        test_dataset = PlantDocDataset(
            cached_data["test"], self.config, split="test")

        return train_dataset, val_dataset, test_dataset

    def setup_model(self):
        """モデルのセットアップ"""
        logger.info("モデルをセットアップしています...")

        try:
            # 必要なクラスのインポート
            from transformers import MllamaForConditionalGeneration, AutoProcessor

            # メモリ設定
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(
                    0).total_memory / 1024**3
                logger.info(f"Total GPU memory: {total_memory:.2f} GB")
                max_memory = {0: f"{int(total_memory * 0.8)}GB"}

            # まずプロセッサーを読み込む
            processor = AutoProcessor.from_pretrained(
                self.config["model"]["name"],
                token=os.getenv("HUGGINGFACE_TOKEN"),
                trust_remote_code=True
            )

            # Vision-Instructモデルの読み込み
            logger.info("Vision-Instructモデルを読み込んでいます...")
            model = MllamaForConditionalGeneration.from_pretrained(
                self.config["model"]["name"],
                token=os.getenv("HUGGINGFACE_TOKEN"),
                device_map="auto",
                torch_dtype=torch.bfloat16,  # モデルの設定に合わせる
                trust_remote_code=True,
                max_memory=max_memory
            )

            # モデル設定の確認
            logger.info(f"Model config: {model.config}")

            # LoRAの設定
            lora_config = LoraConfig(
                r=self.config["training"]["lora"]["rank"],
                lora_alpha=self.config["training"]["lora"]["alpha"],
                target_modules=self.config["training"]["lora"]["target_modules"],
                lora_dropout=self.config["training"]["lora"]["dropout"],
                bias="none",
                task_type="CAUSAL_LM"
            )

            # モデルをLoRA用に準備
            logger.info("LoRAの設定を適用しています...")
            model = model.prepare_for_training(lora_config)

            return model, processor  # tokenizerの代わりにprocessorを返す

        except Exception as e:
            logger.error(f"モデルのロードに失敗: {str(e)}")
            logger.error("詳細なエラー情報: ", exc_info=True)
            raise

    def setup_optimizers(self, model, num_training_steps: int):
        """オプティマイザとスケジューラのセットアップ"""
        optimizer = AdamW(
            model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config["training"]["warmup_steps"],
            num_training_steps=num_training_steps
        )

        return optimizer, scheduler

    def train_epoch(self, model, train_loader, optimizer, scheduler, epoch: int):
        """1エポックの訓練"""
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(progress_bar):
            # バッチの処理
            images = batch["image"].to(self.device)

            # 日本語と英語の両方でトレーニング
            for lang in self.config["data"]["languages"]:
                prompts = batch["prompts"][lang]

                # Forward pass
                outputs = model(
                    images=images,
                    prompts=prompts,
                    labels=batch["label"]
                )

                loss = outputs.loss / len(self.config["data"]["languages"])
                total_loss += loss.item()

                # Backward pass
                self.accelerator.backward(loss)

            # 勾配の累積とアップデート
            if (batch_idx + 1) % self.config["training"]["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config["training"]["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # プログレスバーの更新
            progress_bar.set_postfix({
                "loss": total_loss / (batch_idx + 1),
                "lr": scheduler.get_last_lr()[0]
            })

            # wandbへのログ
            if self.config["wandb"]["log_model"] and batch_idx % self.config["wandb"]["log_interval"] == 0:
                wandb.log({
                    "train_loss": total_loss / (batch_idx + 1),
                    "learning_rate": scheduler.get_last_lr()[0]
                })

        return total_loss / len(train_loader)

    def validate(self, model, val_loader, epoch: int) -> float:
        """検証"""
        model.eval()
        total_loss = 0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Validation {epoch + 1}")
            for batch_idx, batch in enumerate(progress_bar):
                images = batch["image"].to(self.device)

                # 日本語と英語の両方で評価
                for lang in self.config["data"]["languages"]:
                    prompts = batch["prompts"][lang]

                    outputs = model(
                        images=images,
                        prompts=prompts,
                        labels=batch["label"]
                    )

                    loss = outputs.loss / len(self.config["data"]["languages"])
                    total_loss += loss.item()

                progress_bar.set_postfix(
                    {"val_loss": total_loss / (batch_idx + 1)})

        avg_loss = total_loss / len(val_loader)
        if self.config["wandb"]["log_model"]:
            wandb.log({"val_loss": avg_loss})

        return avg_loss

    def save_checkpoint(self, model, epoch: int, loss: float, is_best: bool = False):
        """チェックポイントの保存"""
        checkpoint_dir = Path(self.config["checkpointing"]["output_dir"])
        checkpoint_dir.mkdir(exist_ok=True)

        # 通常のチェックポイント
        checkpoint_path = checkpoint_dir / \
            f"checkpoint_epoch_{epoch}_loss_{loss:.4f}.pt"

        # ベストモデル
        if is_best:
            checkpoint_path = checkpoint_dir / "best_model.pt"

        # モデルの保存
        model.save_pretrained(checkpoint_path)
        logger.info(f"チェックポイントを保存しました: {checkpoint_path}")

    def train(self):
        """トレーニングのメインループ"""
        # データセットの読み込み
        train_dataset, val_dataset, test_dataset = self._load_datasets()

        # データローダーの設定
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"] * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # モデルのセットアップ
        model, processor = self.setup_model()

        # 総ステップの計算
        num_training_steps = len(train_loader) * \
            self.config["training"]["epochs"]

        # オプティマイザとスケジューラのセットアップ
        optimizer, scheduler = self.setup_optimizers(model, num_training_steps)

        # acceleratorによる準備
        model, optimizer, train_loader, val_loader = self.accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )

        best_val_loss = float('inf')
        start_time = time.time()

        logger.info("トレーニングを開始します...")
        for epoch in range(self.config["training"]["epochs"]):
            # トレーニング
            train_loss = self.train_epoch(
                model, train_loader, optimizer, scheduler, epoch)

            # 検証
            val_loss = self.validate(model, val_loader, epoch)

            # ベストモデルの保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(model, epoch, val_loss, is_best=True)

            # 定期的なチェックポイント
            if (epoch + 1) % self.config["checkpointing"]["save_steps"] == 0:
                self.save_checkpoint(model, epoch, val_loss)

            # 進捗の表示
            logger.info(
                f"Epoch {epoch + 1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f}"
            )

        # トレーニング完了
        total_time = time.time() - start_time
        logger.info(f"トレーニングが完了しました。総時間: {total_time / 3600:.2f}時間")


def main():
    parser = argparse.ArgumentParser(description="PlantVisionモデルのトレーニング")
    parser.add_argument("--config", default="config.json", help="設定ファイルのパス")
    parser.add_argument("--data_dir", default="data", help="データディレクトリのパス")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Weights & Biasesを使用する")
    args = parser.parse_args()

    trainer = PlantVisionTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
