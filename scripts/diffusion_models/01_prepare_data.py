import random
import os
import shutil
from tqdm import tqdm

random.seed(42)

train_path = "data/train"
test_path = "data/test"
output_path = "data/musdb18_diffusion"

output_train_path = os.path.join(output_path, "train")
output_valid_path = os.path.join(output_path, "valid")
output_test_path = os.path.join(output_path, "test")

train_filenames = os.listdir(train_path)
random.shuffle(train_filenames)

valid_filenames = train_filenames[int(0.8*len(train_filenames)):]
train_filenames = train_filenames[:int(0.8*len(train_filenames))]
test_filenames = os.listdir(test_path)

if os.path.exists(output_path):
    shutil.rmtree(output_path)


os.makedirs(output_train_path, exist_ok=True)
os.makedirs(output_valid_path, exist_ok=True)
os.makedirs(output_test_path, exist_ok=True)

print(train_filenames)
print(len(valid_filenames))
print(len(test_filenames))

for song_name in tqdm(train_filenames):
    shutil.copytree(os.path.join(train_path, song_name), os.path.join(output_train_path, song_name))
for song_name in tqdm(valid_filenames):
    shutil.copytree(os.path.join(train_path, song_name), os.path.join(output_valid_path, song_name))
for song_name in tqdm(test_filenames):
    shutil.copytree(os.path.join(test_path, song_name), os.path.join(output_test_path, song_name))