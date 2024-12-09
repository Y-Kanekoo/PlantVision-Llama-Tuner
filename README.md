project_root/
├── data/
│   ├── raw/                # PlantDoc-Dataset原本
│   └── processed/          # 前処理済みデータ
├── src/
│   ├── data/
│   │   ├── download.py     # データセットダウンロード
│   │   └── preprocess.py   # データ前処理
│   ├── models/
│   │   ├── trainer.py      # トレーニングロジック
│   │   └── utils.py        # ユーティリティ関数
│   └── eval/
│       └── evaluate.py     # 評価スクリプト
├── configs/
│   └── training_config.yaml # トレーニング設定
├── notebooks/              # 実験用ノートブック
├── outputs/               # モデル出力
├── requirements.txt
└── README.md