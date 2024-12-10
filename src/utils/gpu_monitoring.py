import torch
import psutil
import GPUtil
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt


class GPUMonitor:
    def __init__(self, config, log_interval=1):
        self.config = config
        self.log_interval = log_interval
        self.log_dir = Path(config["output"]["log_dir"]) / "gpu_stats"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stats = []

    def start_monitoring(self):
        """モニタリングの開始"""
        while True:
            try:
                gpu = GPUtil.getGPUs()[0]  # 最初のGPUの情報を取得

                stats = {
                    "timestamp": time.time(),
                    "gpu_load": gpu.load * 100,
                    "gpu_memory_used": gpu.memoryUsed,
                    "gpu_memory_total": gpu.memoryTotal,
                    "gpu_temperature": gpu.temperature,
                    "cpu_percent": psutil.cpu_percent(),
                    "ram_percent": psutil.virtual_memory().percent
                }

                self.stats.append(stats)
                self.save_stats()
                self.plot_stats()

                time.sleep(self.log_interval)

            except KeyboardInterrupt:
                print("Monitoring stopped")
                break

    def save_stats(self):
        """統計情報の保存"""
        with open(self.log_dir / "gpu_stats.json", "w") as f:
            json.dump(self.stats, f, indent=4)

    def plot_stats(self):
        """統計情報のプロット"""
        if len(self.stats) < 2:
            return

        timestamps = [s["timestamp"] - self.stats[0]["timestamp"]
                      for s in self.stats]

        plt.figure(figsize=(15, 10))

        # GPU使用率
        plt.subplot(3, 1, 1)
        plt.plot(timestamps, [s["gpu_load"] for s in self.stats])
        plt.title("GPU Utilization")
        plt.ylabel("Utilization (%)")

        # GPUメモリ使用量
        plt.subplot(3, 1, 2)
        plt.plot(timestamps, [s["gpu_memory_used"] for s in self.stats])
        plt.title("GPU Memory Usage")
        plt.ylabel("Memory (MB)")

        # GPU温度
        plt.subplot(3, 1, 3)
        plt.plot(timestamps, [s["gpu_temperature"] for s in self.stats])
        plt.title("GPU Temperature")
        plt.ylabel("Temperature (°C)")
        plt.xlabel("Time (s)")

        plt.tight_layout()
        plt.savefig(self.log_dir / "gpu_stats.png")
        plt.close()
