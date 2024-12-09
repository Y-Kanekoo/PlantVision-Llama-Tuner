# Plant Vision Llama Tuner

このプロジェクトは、Llama 3.2 11B Vision-Instructモデルを用いて、PlantDoc-Datasetを対象に植物疾患分類タスクのファインチューニングを行うものです。本リポジトリには、環境構築方法、データセットの前処理、トレーニング手順、評価方法、モデル出力物の格納先などを詳細にまとめています。また、`unsloth`を用いてトレーニング高速化を試みており、mixed precisionやgradient checkpointing、GPU最適化なども検討しています。

以下では、プロジェクトの設計から実行方法、環境構築、データ処理、トレーニングと評価の流れを詳細に説明し、最後に追加で確認したい不明点や前提条件も提示します。

## プロジェクト概要

- **モデル**: [Llama 3.2 11B Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
- **データセット**: [PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset.git)  
  - 植物疾患分類タスク向けのデータセット
- **タスク**: 画像入力＋テキスト指示に基づく疾病分類 (Vision + Text Instruct)
- **高速化ツール**: [unsloth](https://github.com/unslothai/unsloth)
- **環境**:
  - ローカルマシン上で実行
  - Python 3.10 & CUDA 12.1
  - GPU: NVIDIA GeForce RTX 4070 Ti
  - 仮想環境: `env` ディレクトリに作成
- **パッケージ管理**: `requirements.txt` による管理

## ディレクトリ構成

PlantVision Llama Tuner/ ├─ configs/ # 設定ファイル（ハイパーパラメータ、学習設定、評価設定など） ├─ data/ # PlantDoc-Datasetや前処理済みデータを格納 ├─ env/ # Python仮想環境用ディレクトリ (venv) ├─ notebooks/ # Jupyter Notebook形式での実験・可視化・デバッグ用 ├─ outputs/ # 学習済みモデル、チェックポイント、ログ、評価結果、図表などを格納 ├─ src/ # ソースコード（トレーニングスクリプト、データ前処理スクリプト、推論用コードなど） ├─ requirements.txt # 必要なライブラリ定義 ├─ README.md # 本ファイル └─ .gitignore