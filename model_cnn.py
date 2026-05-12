"""
STEP 3: Deep Learning Model — CNN
- Data augmentation (auto-disabled during validation/inference)
- 3-block CNN with BatchNormalization
- EarlyStopping + ReduceLROnPlateau callbacks
- Higher capacity: Dense(256) with Dropout(0.5)
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns

CLASSES    = ['rock', 'paper', 'scissors']
MODEL_PATH = 'model_cnn.keras'

print("=" * 40)
print("  Training CNN Model (Enhanced)...")
print("=" * 40)

# Load data
X_train = np.load('X_train.npy')
X_test  = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test  = np.load('y_test.npy')

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── Build CNN ──────────────────────────────────────────
# Augmentation layers placed first in Sequential are automatically
# active only during training — Keras passes training=False for
# evaluate/predict, so validation metrics are never corrupted.
model = models.Sequential([
    # Augmentation (training only)
    layers.RandomFlip("horizontal", input_shape=(64, 64, 1)),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),

    # Block 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Block 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Block 3
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Fully connected
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Callbacks ──────────────────────────────────────────
cbs = [
    callbacks.EarlyStopping(
        monitor='val_accuracy', patience=8,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=4, min_lr=1e-6, verbose=1
    ),
]

# ── Train ──────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=cbs
)

# ── Evaluate ───────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test), axis=1)

print(f"\n=== CNN Results ===")
print(f"Test Accuracy: {acc * 100:.2f}%")
print(f"Test Loss:     {loss:.4f}")

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=[0, 1, 2]
)
print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
print("-" * 55)
for i, cls in enumerate(CLASSES):
    print(f"{cls:<12} {precision[i]*100:>9.1f}% {recall[i]*100:>9.1f}%"
          f" {f1[i]*100:>9.1f}% {int(support[i]):>10}")
print("-" * 55)
print(f"{'Overall':<12} {acc*100:>9.1f}%")

# ── Training Curves ────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'CNN Training Results  —  Final Accuracy: {acc*100:.2f}%',
             fontsize=15, fontweight='bold', y=1.02)

epochs_ran  = range(1, len(history.history['accuracy']) + 1)
acc_pct     = [v * 100 for v in history.history['accuracy']]
val_acc_pct = [v * 100 for v in history.history['val_accuracy']]

ax1.plot(epochs_ran, acc_pct,     'o-', color='#2196F3', linewidth=2, markersize=4, label='Train')
ax1.plot(epochs_ran, val_acc_pct, 's-', color='#FF5722', linewidth=2, markersize=4, label='Validation')
ax1.set_title('Accuracy per Epoch', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_ylim(0, 105)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax1.legend(fontsize=11)
ax1.tick_params(labelsize=10)

ax2.plot(epochs_ran, history.history['loss'],     'o-', color='#2196F3', linewidth=2, markersize=4, label='Train')
ax2.plot(epochs_ran, history.history['val_loss'], 's-', color='#FF5722', linewidth=2, markersize=4, label='Validation')
ax2.set_title('Loss per Epoch', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.legend(fontsize=11)
ax2.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Training curves saved: training_curves.png")

# ── Confusion Matrix ───────────────────────────────────
cm      = confusion_matrix(y_test, y_pred)
cm_pct  = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
# Cell labels: count on top, percentage below
annot   = np.array([[f"{cm[i,j]}\n({cm_pct[i,j]:.0f}%)"
                      for j in range(3)] for i in range(3)])

plt.figure(figsize=(7, 6))
sns.heatmap(cm_pct, annot=annot, fmt='', cmap='Blues', linewidths=0.5,
            annot_kws={'size': 12, 'weight': 'bold'},
            xticklabels=CLASSES, yticklabels=CLASSES,
            vmin=0, vmax=100)
plt.title(f'CNN — Confusion Matrix  ({acc*100:.2f}% accuracy)',
          fontsize=14, fontweight='bold', pad=12)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('confusion_matrix_cnn.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix saved: confusion_matrix_cnn.png")

# ── Save Model ─────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\n✅ CNN model saved: {MODEL_PATH}")
