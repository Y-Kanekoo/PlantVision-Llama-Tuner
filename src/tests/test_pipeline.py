import unittest
import torch
import yaml
from pathlib import Path
from src.data.preprocess import PlantDocPreprocessor
from src.data.dataloader import get_dataloaders
from src.models.trainer import PlantDocTrainer
from src.eval.evaluate import PlantDocEvaluator


class TestPlantDocPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """テストの初期設定"""
        # 設定ファイルの読み込み
        with open("configs/training_config.yaml", "r") as f:
            cls.config = yaml.safe_load(f)

        # テスト用に設定を変更
        cls.config["training"]["batch_size"] = 2
        cls.config["training"]["num_epochs"] = 1

    def test_data_preprocessing(self):
        """データ前処理のテスト"""
        preprocessor = PlantDocPreprocessor(self.config)
        train_df, val_df, test_df = preprocessor.process_dataset()

        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(val_df), 0)
        self.assertGreater(len(test_df), 0)

    def test_model_forward_pass(self):
        """モデル���順伝播テスト"""
        trainer = PlantDocTrainer(self.config)

        # ダミーデータの作成
        dummy_batch = {
            "image": torch.randn(2, 3, 224, 224),
            "label": torch.tensor([0, 1])
        }

        # GPUが利用可能な場合はGPUに移動
        if torch.cuda.is_available():
            dummy_batch = {k: v.cuda() for k, v in dummy_batch.items()}

        outputs = trainer.model(
            pixel_values=dummy_batch["image"],
            labels=dummy_batch["label"]
        )

        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.logits.shape[0], 2)

    def test_memory_usage(self):
        """メモリ使用量のテスト"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

            # モデルの初期化とダミーデータでの学習
            trainer = PlantDocTrainer(self.config)
            dummy_batch = {
                "image": torch.randn(2, 3, 224, 224).cuda(),
                "label": torch.tensor([0, 1]).cuda()
            }

            outputs = trainer.model(
                pixel_values=dummy_batch["image"],
                labels=dummy_batch["label"]
            )
            loss = outputs.loss
            loss.backward()

            max_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Peak GPU memory usage: {max_memory:.2f} MB")

            self.assertLess(max_memory, 11000)  # 11GB以下であることを確認


if __name__ == "__main__":
    unittest.main()
