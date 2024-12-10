import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np


class ModelOptimizer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.scaler = GradScaler()

    def find_optimal_batch_size(self, sample_input):
        """最適なバッチサイズの探索"""
        batch_sizes = [2, 4, 8, 16, 32]
        memory_usage = []

        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            try:
                # バッチサイズを変更してテスト
                test_input = {
                    "image": sample_input["image"].repeat(batch_size, 1, 1, 1),
                    "label": sample_input["label"].repeat(batch_size)
                }

                with autocast():
                    outputs = self.model(**test_input)
                    loss = outputs.loss
                    self.scaler.scale(loss).backward()

                max_memory = torch.cuda.max_memory_allocated() / 1024**2
                memory_usage.append(max_memory)
                torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"Error at batch size {batch_size}: {e}")
                break

        # 最適なバッチサイズの選択
        valid_batch_sizes = batch_sizes[:len(memory_usage)]
        optimal_idx = np.argmin(memory_usage)
        return valid_batch_sizes[optimal_idx]

    def optimize_learning_rate(self, train_loader, val_loader):
        """学習率の最適化"""
        lr_finder = torch.optim.lr_finder.LRFinder(
            self.model,
            self.optimizer,
            torch.nn.CrossEntropyLoss(),
            device=self.device
        )

        lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
        lr_finder.plot()  # 学習率vs損失のプロット
        lr_finder.reset()  # モデルとオプティマイザをリセット

        return lr_finder.suggestion()
