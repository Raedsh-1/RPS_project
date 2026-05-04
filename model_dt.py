"""
STEP 2: Machine Learning Model — Decision Tree (Baseline)
- Loads preprocessed data
- Flattens images into 1D vectors
- Trains a Decision Tree classifier
- Prints accuracy, precision, recall, F1-score
- Saves the model as model_dt.pkl
"""

import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CLASSES = ['rock', 'paper', 'scissors']

print("=" * 40)
print("  Training Decision Tree Model...")
print("=" * 40)

# Load data
X_train = np.load('X_train.npy')
X_test  = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test  = np.load('y_test.npy')

# Flatten images: (N, 64, 64, 3) -> (N, 12288)
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat  = X_test.reshape(len(X_test), -1)

print(f"Train samples: {X_train_flat.shape[0]}")
print(f"Test  samples: {X_test_flat.shape[0]}")
print(f"Features per image: {X_train_flat.shape[1]}")

# Train
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_flat, y_train)

# Evaluate
y_pred = dt.predict(X_test_flat)

print("\n=== Decision Tree Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASSES))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Decision Tree — Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_dt.png')
plt.show()
print("Confusion matrix saved: confusion_matrix_dt.png")

# Save model
with open('model_dt.pkl', 'wb') as f:
    pickle.dump(dt, f)
print("\n✅ Decision Tree model saved: model_dt.pkl")
