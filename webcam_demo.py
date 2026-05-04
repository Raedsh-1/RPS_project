"""
STEP 4: Live Webcam Demo
- Opens webcam
- Detects hand inside the green box
- Predicts: Rock / Paper / Scissors
- Shows confidence % for each class
- Press 'q' to quit
"""

import cv2
import numpy as np
import tensorflow as tf

# ── Config ─────────────────────────────────────────
CLASSES   = ['Rock', 'Paper', 'Scissors']
IMG_SIZE  = 64
MODEL_PATH = 'model_cnn.h5'

# Box region where user places their hand
BOX_X1, BOX_Y1 = 100, 100
BOX_X2, BOX_Y2 = 350, 350

# ── Load Model ─────────────────────────────────────
print("Loading CNN model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded. Opening webcam...")

# ── Webcam ─────────────────────────────────────────
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam. Check your camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Extract ROI (Region of Interest)
    roi = frame[BOX_Y1:BOX_Y2, BOX_X1:BOX_X2]

    # Preprocess ROI
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img_input = img.astype('float32') / 255.0
    img_input = np.expand_dims(img_input, axis=0)  # (1, 64, 64, 3)

    # Predict
    probs = model.predict(img_input, verbose=0)[0]
    pred_idx   = np.argmax(probs)
    pred_label = CLASSES[pred_idx]
    pred_conf  = probs[pred_idx] * 100

    # ── Draw UI ────────────────────────────────────
    # Green box
    cv2.rectangle(frame, (BOX_X1, BOX_Y1), (BOX_X2, BOX_Y2), (0, 255, 0), 2)
    cv2.putText(frame, "Place hand here",
                (BOX_X1, BOX_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Main prediction
    cv2.putText(frame, f"Prediction: {pred_label}  ({pred_conf:.1f}%)",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

    # All class probabilities
    for i, (cls, prob) in enumerate(zip(CLASSES, probs)):
        bar_len = int(prob * 200)
        color = (0, 255, 100) if i == pred_idx else (100, 100, 100)
        cv2.rectangle(frame,
                      (10, 70 + i * 35),
                      (10 + bar_len, 90 + i * 35),
                      color, -1)
        cv2.putText(frame, f"{cls}: {prob*100:.1f}%",
                    (220, 88 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    cv2.putText(frame, "Press 'q' to quit",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Rock Paper Scissors — AI Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
