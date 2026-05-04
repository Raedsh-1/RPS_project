# 🎮 Rock Paper Scissors — AI Project
### Academic Year 2025-2026 | Deep Learning + Computer Vision

---

## 📁 Project Structure

```
RPS_Project/
│
├── dataset/
│   ├── rock/           ← put rock images here
│   ├── paper/          ← put paper images here
│   └── scissors/       ← put scissors images here
│
├── preprocess.py       ← Step 2: Load & prepare data
├── model_dt.py         ← Step 3: Decision Tree (ML baseline)
├── model_cnn.py        ← Step 4: CNN (Deep Learning)
├── webcam_demo.py      ← Step 5: Live webcam demo
│
├── requirements.txt    ← All required libraries
└── README.md           ← This file
```

---

## ⚙️ Setup (Do This Once)

### 1. Open VS Code → open this folder → press Ctrl+` to open terminal

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate it
```bash
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```
You should see **(venv)** at the start of the terminal line.

### 4. Install all libraries
```bash
pip install -r requirements.txt
```

---

## 📥 Download the Dataset

1. Go to: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors
2. Click **Download**
3. Unzip the file
4. Copy the images into the correct folders:
   - rock images → `dataset/rock/`
   - paper images → `dataset/paper/`
   - scissors images → `dataset/scissors/`

---

## ▶️ Run the Project (In Order)

### Step 1 — Preprocess the data
```bash
python preprocess.py
```
This creates: `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`

---

### Step 2 — Train Decision Tree (baseline ML model)
```bash
python model_dt.py
```
This creates: `model_dt.pkl`, `confusion_matrix_dt.png`

---

### Step 3 — Train CNN (deep learning model)
```bash
python model_cnn.py
```
This creates: `model_cnn.h5`, `training_curves.png`, `confusion_matrix_cnn.png`

---

### Step 4 — Run Live Webcam Demo
```bash
python webcam_demo.py
```
- A window opens with your webcam feed
- Place your hand inside the **green box**
- The AI predicts: Rock / Paper / Scissors
- Confidence % shown for each class
- Press **Q** to quit

---

## 📊 Expected Results

| Model         | Expected Accuracy |
|---------------|-------------------|
| Decision Tree | ~60–75%           |
| CNN           | ~90–98%           |

The CNN performs much better because it learns spatial features from images.

---

## 👥 Team
- Student 1: _______________
- Student 2: _______________

Academic Year: 2025–2026
