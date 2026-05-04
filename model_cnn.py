"""
STEP 3: Deep Learning Model — CNN
- Builds a Convolutional Neural Network
- Trains for 20 epochs, batch size 32, optimizer Adam
- Plots accuracy and loss curves
- Prints classification report
- Saves model as model_cnn.h5
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

CLASSES = ['rock', 'paper', 'scissors']

print("=" * 40)
print("  Training CNN Model...")
print("=" * 40)

# Load data
X_train = np.load('X_train.npy')
X_test  = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test  = np.load('y_test.npy')

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── Build CNN ──────────────────────────────────────
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2, 2),

    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # Fully connected
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    # Output: 3 classes
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Train ──────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ── Evaluate ───────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test), axis=1)

print(f"\n=== CNN Results ===")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Loss:     {loss:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASSES))

# ── Training Curves ────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'],     label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'],     label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
print("Training curves saved: training_curves.png")

# ── Confusion Matrix ───────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('CNN — Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_cnn.png')
plt.show()
print("Confusion matrix saved: confusion_matrix_cnn.png")

# ── Save Model ─────────────────────────────────────
model.save('model_cnn.h5')
print("\n✅ CNN model saved: model_cnn.h5")
