# Download the images
files.upload()

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!pip install kaggle
!kaggle datasets download -d michaeljkerr/20k-album-covers-within-20-genres
!unzip 20k-album-covers-within-20-genres.zip -d album_covers

# Delete death metal and doom metal
import os
import shutil
from random import shuffle
shutil.rmtree('album_covers/GAID/DeathMetal')
shutil.rmtree('album_covers/GAID/DoomMetal')

# Create train, test, and val folders
base_dir = 'album_covers/GAID'

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

def is_valid_dir(d):
    return not d.startswith('.') and os.path.isdir(os.path.join(base_dir, d))
categories = [d for d in os.listdir(base_dir) if is_valid_dir(d)]

for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

train_ratio = 0.8
val_ratio = 0.1

for category in categories:
    category_path = os.path.join(base_dir, category)
    files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
    shuffle(files)  # Shuffle to ensure randomness

    total_files = len(files)
    train_end = int(train_ratio * total_files)
    val_end = train_end + int(val_ratio * total_files)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    for f in train_files:
        shutil.move(os.path.join(category_path, f), os.path.join(train_dir, category, f))
    for f in val_files:
        shutil.move(os.path.join(category_path, f), os.path.join(val_dir, category, f))
    for f in test_files:
        shutil.move(os.path.join(category_path, f), os.path.join(test_dir, category, f))

# Delete old folders
shutil.rmtree('album_covers/GAID/Blues')
shutil.rmtree('album_covers/GAID/Classical')
shutil.rmtree('album_covers/GAID/Country')
shutil.rmtree('album_covers/GAID/DrumNBass')
shutil.rmtree('album_covers/GAID/Electronic')
shutil.rmtree('album_covers/GAID/Folk')
shutil.rmtree('album_covers/GAID/Grime')
shutil.rmtree('album_covers/GAID/HeavyMetal')
shutil.rmtree('album_covers/GAID/HipHop')
shutil.rmtree('album_covers/GAID/Jazz')
shutil.rmtree('album_covers/GAID/LoFi')
shutil.rmtree('album_covers/GAID/Pop')
shutil.rmtree('album_covers/GAID/PsychedelicRock')
shutil.rmtree('album_covers/GAID/Punk')
shutil.rmtree('album_covers/GAID/Reggae')
shutil.rmtree('album_covers/GAID/Rock')
shutil.rmtree('album_covers/GAID/Soul')
shutil.rmtree('album_covers/GAID/Techno')

shutil.rmtree('album_covers/GAID/test/test')
shutil.rmtree('album_covers/GAID/test/train')
shutil.rmtree('album_covers/GAID/test/val')

shutil.rmtree('album_covers/GAID/train/test')
shutil.rmtree('album_covers/GAID/train/train')
shutil.rmtree('album_covers/GAID/train/val')

shutil.rmtree('album_covers/GAID/val/test')
shutil.rmtree('album_covers/GAID/val/train')
shutil.rmtree('album_covers/GAID/val/val')
