import torch
from transformers import AutoProcessor, LlamaForCausalLM
from unsloth import FastLlamaModel
from tqdm import tqdm
import logging
from pathlib import Path


class PlantDocTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.setup_logging()
        self.setup_model()

    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    Path(self.config["output"]["log_dir"]) / "training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_model(self):
        """モデルのセットアップ"""
        # Llamaモデルの初期化とunslothによる最適化
        self.model = FastLlamaModel.from_pretrained(
            model_name=self.config["model"]["name"],
            max_seq_length=self.config["model"]["max_length"],
            dtype="bfloat16",
            load_in_4bit=True,  # 4bit量子化で省メモリ化
            device_map="auto",  # 自動デバイス割り当て
        )

        # プロセッサーの設定
        self.processor = AutoProcessor.from_pretrained(
            self.config["model"]["name"],
            trust_remote_code=True
        )

        # オプティマイザの設定
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"]
        )

    def train_epoch(self, train_loader):
        """1エポックの学習"""
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=self.processor(images, return_tensors="pt")[
                    "input_ids"].to(self.device),
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """検証"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=self.processor(images, return_tensors="pt")[
                        "input_ids"].to(self.device),
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                # 精度の計算
                predictions = outputs.logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        return total_loss / len(val_loader), accuracy

    # 他のメソッドは変更なし
