"""
STEP 1: Data Preprocessing
- Loads images from dataset/rock, dataset/paper, dataset/scissors
- Resizes to 64x64
- Normalizes pixel values
- Splits into 80% train / 20% test
- Saves as .npy files
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

IMG_SIZE = 64
CLASSES = ['rock', 'paper', 'scissors']
DATA_DIR = 'dataset'

def load_data():
    X, y = [], []
    for label, cls in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, cls)
        if not os.path.exists(folder):
            print(f"[WARNING] Folder not found: {folder}")
            continue
        files = os.listdir(folder)
        print(f"Loading '{cls}': {len(files)} images...")
        for img_name in files:
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)
    return np.array(X), np.array(y)

print("=" * 40)
print("  Loading and preprocessing data...")
print("=" * 40)

X, y = load_data()
print(f"\nTotal images loaded: {len(X)}")

# Normalize pixel values to [0, 1]
X = X.astype('float32') / 255.0

# Train/Test split: 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save arrays
np.save('X_train.npy', X_train)
np.save('X_test.npy',  X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy',  y_test)

print(f"\nTrain set: {X_train.shape}")
print(f"Test  set: {X_test.shape}")
print("\n✅ Data saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
