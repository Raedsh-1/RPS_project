"""
STEP 1: Data Preprocessing
- Converts each image to a Canny edge map (removes background colour dependency)
- Resizes to 64x64, normalizes, splits 80/20
- Saves as .npy files
- Shows sample images and class distribution chart
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

IMG_SIZE = 64
CLASSES  = ['rock', 'paper', 'scissors']
DATA_DIR = 'dataset'

def apply_edges(img_bgr):
    """Convert a BGR image to a Canny edge map (background-independent)."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges   # shape: (H, W) — 1 channel, values 0 or 255

def load_data():
    X, y = [], []
    samples_orig  = {cls: [] for cls in CLASSES}
    samples_edges = {cls: [] for cls in CLASSES}

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
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_edges   = apply_edges(img_resized)
            X.append(img_edges)
            y.append(label)
            if len(samples_orig[cls]) < 5:
                samples_orig[cls].append(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
                samples_edges[cls].append(img_edges)

    return np.array(X), np.array(y), samples_orig, samples_edges

print("=" * 40)
print("  Loading and preprocessing data...")
print("=" * 40)

X, y, samples_orig, samples_edges = load_data()
print(f"\nTotal images loaded: {len(X)}")

# Normalize to [0, 1] and add channel dimension → (N, 64, 64, 1)
X = X.astype('float32') / 255.0
X = np.expand_dims(X, axis=-1)

# Train/Test split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

np.save('X_train.npy', X_train)
np.save('X_test.npy',  X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy',  y_test)

print(f"\nTrain set : {X_train.shape}  ({len(y_train)} images)")
print(f"Test  set : {X_test.shape}   ({len(y_test)} images)")

# ── Class distribution chart ───────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')

counts      = [int(np.sum(y == i)) for i in range(len(CLASSES))]
bar_colors  = ['#7EC87A', '#6BAED6', '#FDAE6B']
label_names = [c.capitalize() for c in CLASSES]
total       = sum(counts)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(label_names, counts, color=bar_colors,
              edgecolor='#333333', linewidth=1.2, width=0.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{count}\n({count/total*100:.1f}%)",
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Dataset — Class Distribution', fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('Number of Images', fontsize=12)
ax.set_xlabel('Class', fontsize=12)
ax.set_ylim(0, max(counts) * 1.18)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Class distribution chart saved: class_distribution.png")

# ── Sample images: original vs edge ───────────────────
n_cols = 5
fig, axes = plt.subplots(len(CLASSES) * 2, n_cols,
                         figsize=(n_cols * 2.2, len(CLASSES) * 4.5))
fig.suptitle('Dataset — Original vs Edge Detection (what the model sees)',
             fontsize=13, fontweight='bold', y=1.01)

for row, cls in enumerate(CLASSES):
    for col in range(n_cols):
        ax_orig  = axes[row * 2][col]
        ax_edges = axes[row * 2 + 1][col]

        if col < len(samples_orig[cls]):
            ax_orig.imshow(samples_orig[cls][col])
            ax_edges.imshow(samples_edges[cls][col], cmap='gray')
        ax_orig.axis('off')
        ax_edges.axis('off')

        if col == 0:
            ax_orig.set_title(f'{cls.capitalize()}\n(original)',
                              fontsize=10, fontweight='bold', loc='left')
            ax_edges.set_title('(edges)', fontsize=9,
                               color='#555555', loc='left')

plt.tight_layout()
plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
plt.show()
print("Sample images saved: sample_images.png")

print("\n✅ Data saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
