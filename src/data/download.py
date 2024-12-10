import os
import requests
import git
from pathlib import Path


def download_plantdoc_dataset():
    """
    PlantDoc-Datasetをダウンロードし、data/rawディレクトリに保存
    """
    dataset_url = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
    raw_data_dir = Path("data/raw")

    if not raw_data_dir.exists():
        raw_data_dir.mkdir(parents=True)
        git.Repo.clone_from(dataset_url, raw_data_dir)
