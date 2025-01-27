from pathlib import Path
import os
import shutil
import random

random.seed(42)

dataset_path = Path("data/musdb18_hq")

if not os.path.exists(dataset_path / "validation"):
    os.makedirs(dataset_path / "validation", exist_ok=True)

    train_songs = [song.name for song in (dataset_path / "train").iterdir()]
    val_songs = random.sample(train_songs, 20)

    for song in val_songs:
        shutil.move(os.path.join(dataset_path, "train", song), dataset_path / "validation")