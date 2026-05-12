"""
STEP 4: Live Webcam Demo
- Applies the same Canny edge preprocessing used during training
- Predicts Rock / Paper / Scissors from hand shape (background-independent)
- Shows edge preview inside the green box so you can see what the model sees
- Smooths predictions over last 10 frames
- Press 'q' to quit
"""

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time

# ── Config ─────────────────────────────────────────────
CLASSES        = ['Rock', 'Paper', 'Scissors']
IMG_SIZE       = 64
MODEL_PATH     = 'model_cnn.keras'
SMOOTH_FRAMES  = 10
CONF_THRESHOLD = 0.60

BOX_X1, BOX_Y1 = 100, 100
BOX_X2, BOX_Y2 = 350, 350

def apply_edges(img_bgr):
    """Same preprocessing used in preprocess.py — must stay identical."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

# ── Load Model ─────────────────────────────────────────
print("Loading CNN model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded. Opening webcam...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("Press 'q' to quit.")

prob_buffer = deque(maxlen=SMOOTH_FRAMES)
prev_time   = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)

    # ── FPS ────────────────────────────────────────────
    now       = time.time()
    fps       = 1.0 / (now - prev_time + 1e-9)
    prev_time = now

    # ── ROI → edge map → predict ───────────────────────
    roi        = frame[BOX_Y1:BOX_Y2, BOX_X1:BOX_X2]
    roi_64     = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    edges      = apply_edges(roi_64)                        # (64, 64)
    img_input  = edges.astype('float32') / 255.0
    img_input  = img_input[np.newaxis, :, :, np.newaxis]   # (1, 64, 64, 1)

    probs        = model.predict(img_input, verbose=0)[0]
    prob_buffer.append(probs)
    smooth_probs = np.mean(prob_buffer, axis=0)
    pred_idx     = int(np.argmax(smooth_probs))
    pred_conf    = smooth_probs[pred_idx]

    # ── Show edge preview inside the green box ─────────
    # Resize edge map back to the box size so user sees what the model sees
    edge_preview = cv2.resize(edges, (BOX_X2 - BOX_X1, BOX_Y2 - BOX_Y1))
    edge_color   = cv2.cvtColor(edge_preview, cv2.COLOR_GRAY2BGR)
    # Blend 40% edge / 60% original so the hand is still visible
    roi_blend    = cv2.addWeighted(roi, 0.6, edge_color, 0.4, 0)
    frame[BOX_Y1:BOX_Y2, BOX_X1:BOX_X2] = roi_blend

    # ── Draw UI ────────────────────────────────────────
    cv2.rectangle(frame, (BOX_X1, BOX_Y1), (BOX_X2, BOX_Y2), (0, 255, 0), 2)
    cv2.putText(frame, "Place hand here",
                (BOX_X1, BOX_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    # Main prediction
    if pred_conf >= CONF_THRESHOLD:
        label_text  = f"{CLASSES[pred_idx]}  {pred_conf*100:.1f}%"
        label_color = (0, 200, 255)
    else:
        label_text  = "Waiting..."
        label_color = (160, 160, 160)

    cv2.putText(frame, label_text,
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 2)

    # Confidence bars
    bar_max_w = 180
    for i, (cls, prob) in enumerate(zip(CLASSES, smooth_probs)):
        bar_y1 = 75 + i * 38
        bar_y2 = bar_y1 + 22
        bar_w  = int(prob * bar_max_w)
        cv2.rectangle(frame, (10, bar_y1), (10 + bar_max_w, bar_y2), (50, 50, 50), -1)
        color = (0, 220, 80) if i == pred_idx else (100, 100, 180)
        cv2.rectangle(frame, (10, bar_y1), (10 + bar_w, bar_y2), color, -1)
        cv2.putText(frame, f"{cls}: {prob*100:.1f}%",
                    (200, bar_y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (frame.shape[1] - 110, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(frame, "Press 'q' to quit",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Rock Paper Scissors — AI Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
