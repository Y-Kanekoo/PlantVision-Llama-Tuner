import torch
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class PlantDocEvaluator:
    def __init__(self, model, config, processor, device):
        self.model = model
        self.config = config
        self.processor = processor
        self.device = device
        self.setup_output_dir()

    def setup_output_dir(self):
        """出力ディレクトリの設定"""
        self.output_dir = Path(self.config["output"]["log_dir"]) / "evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, test_loader, class_names):
        """評価の実行"""
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(pixel_values=images)
                preds = outputs.logits.argmax(dim=-1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # 評価指標の計算と保存
        self.save_classification_report(predictions, true_labels, class_names)
        self.plot_confusion_matrix(predictions, true_labels, class_names)

    def save_classification_report(self, predictions, true_labels, class_names):
        """分類レポートの保存"""
        report = classification_report(
            true_labels,
            predictions,
            target_names=class_names,
            output_dict=True
        )

        # JSONとして保存
        with open(self.output_dir / "classification_report.json", "w") as f:
            json.dump(report, f, indent=4)

    def plot_confusion_matrix(self, predictions, true_labels, class_names):
        """混同行列のプロット"""
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(15, 15))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png")
        plt.close()
