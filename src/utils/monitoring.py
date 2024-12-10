import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd


class TrainingMonitor:
    def __init__(self, config):
        self.config = config
        self.log_dir = Path(config["output"]["log_dir"])
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }

    def update(self, train_loss, val_loss, val_accuracy):
        """学習履歴の更新"""
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_accuracy"].append(val_accuracy)

    def plot_training_history(self):
        """学習履歴のプロット"""
        plt.figure(figsize=(12, 4))

        # Loss のプロット
        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Accuracy の���ロット
        plt.subplot(1, 2, 2)
        plt.plot(self.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.log_dir / "training_history.png")
        plt.close()

    def save_history(self):
        """学習履歴の保存"""
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.log_dir / "training_history.csv", index=False)

        with open(self.log_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=4)
