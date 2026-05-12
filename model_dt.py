"""
STEP 2: Machine Learning Model — Decision Tree (Baseline)
- Loads preprocessed data
- Flattens images into 1D vectors
- Finds best hyperparameters via GridSearchCV
- Prints accuracy as percentage
- Saves confusion matrix and oval-node tree diagram
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

CLASSES = ['Rock', 'Paper', 'Scissors']

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
X_test_flat  = X_test.reshape(len(X_test),  -1)

print(f"Train samples:      {X_train_flat.shape[0]}")
print(f"Test  samples:      {X_test_flat.shape[0]}")
print(f"Features per image: {X_train_flat.shape[1]}")

# ── Hyperparameter Search ──────────────────────────────
print("\nSearching best hyperparameters (this may take a minute)...")

param_grid = {
    'max_depth':         [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'criterion':         ['gini', 'entropy'],
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train_flat, y_train)

print(f"Best params:        {grid_search.best_params_}")
print(f"Best CV F1 (macro): {grid_search.best_score_ * 100:.2f}%")

# ── Evaluate ───────────────────────────────────────────
dt     = grid_search.best_estimator_
y_pred = dt.predict(X_test_flat)
acc    = accuracy_score(y_test, y_pred)

print("\n=== Decision Tree Results ===")
print(f"Test Accuracy: {acc * 100:.2f}%")

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

# ── Confusion Matrix ───────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', linewidths=0.5,
            annot_kws={'size': 14, 'weight': 'bold'},
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Decision Tree — Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('confusion_matrix_dt.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix saved: confusion_matrix_dt.png")

# ── Custom Oval-Node Tree Diagram ─────────────────────
def plot_oval_tree(clf, class_names, max_depth=3, figsize=(26, 12)):
    """
    Draw a presentation-ready decision tree:
    - Internal nodes: white oval  — show image region, colour channel, threshold, sample count
    - Leaf nodes:     colour oval — colour-coded by predicted class
    - Arrows:         True (left) / False (right) labels
    """
    tree_ = clf.tree_
    LEAF  = _tree.TREE_LEAF
    IMG   = 64   # image size used in preprocessing

    # Leaf colours match class identity
    CLASS_FACE   = ['#7EC87A', '#6BAED6', '#FDAE6B']   # Rock=green, Paper=blue, Scissors=orange
    CLASS_BORDER = ['#2E7D32', '#1565C0', '#BF360C']

    INT_FACE   = '#FFFFFF'   # internal nodes: white
    INT_BORDER = '#555555'
    ARROW_CLR  = '#444444'

    def decode_feature(feat_idx):
        """Turn a flat pixel index into a human-readable location + channel."""
        ch_name = ['Red', 'Green', 'Blue'][feat_idx % 3]
        row = feat_idx // (IMG * 3)
        col = (feat_idx % (IMG * 3)) // 3
        r_zone = 'Top'    if row < IMG // 3 else ('Middle' if row < 2 * IMG // 3 else 'Bottom')
        c_zone = 'Left'   if col < IMG // 3 else ('Center' if col < 2 * IMG // 3 else 'Right')
        return f"{r_zone}-{c_zone}  •  {ch_name}"

    # ── Layout ────────────────────────────────────────
    def count_leaves(node, depth):
        if depth >= max_depth or tree_.children_left[node] == LEAF:
            return 1
        return (count_leaves(tree_.children_left[node],  depth + 1) +
                count_leaves(tree_.children_right[node], depth + 1))

    node_pos = {}

    def assign_pos(node, depth, x0, x1):
        node_pos[node] = ((x0 + x1) / 2.0, -depth * 2.8)
        if depth < max_depth and tree_.children_left[node] != LEAF:
            l, r   = tree_.children_left[node], tree_.children_right[node]
            ll, rl = count_leaves(l, depth + 1), count_leaves(r, depth + 1)
            split  = x0 + (x1 - x0) * ll / (ll + rl)
            assign_pos(l, depth + 1, x0,    split)
            assign_pos(r, depth + 1, split, x1)

    total_w = count_leaves(0, 0) * 4.0
    assign_pos(0, 0, 0, total_w)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('#F7F9FC')
    ax.axis('off')

    # ── Draw ─────────────────────────────────────────
    def draw(node, depth, parent=None, is_left=None):
        if node not in node_pos:
            return
        x, y = node_pos[node]

        is_leaf   = (depth >= max_depth or tree_.children_left[node] == LEAF)
        n_samples = int(tree_.n_node_samples[node])
        cls_idx   = int(np.argmax(tree_.value[node]))

        # ── Arrow + True/False label ──────────────────
        if parent is not None:
            px, py = node_pos[parent]
            ax.annotate(
                '', xy=(x, y + 0.48), xytext=(px, py - 0.48),
                arrowprops=dict(arrowstyle='->', color=ARROW_CLR,
                                lw=2.0, mutation_scale=16),
                zorder=2
            )
            lbl = 'True' if is_left else 'False'
            mx, my = (x + px) / 2, (y + py) / 2
            ax.text(mx, my, lbl,
                    fontsize=10, ha='center', va='center', fontstyle='italic',
                    color='#333333',
                    bbox=dict(facecolor='#F7F9FC', edgecolor='none', pad=2),
                    zorder=3)

        # ── Node appearance ───────────────────────────
        if is_leaf:
            face   = CLASS_FACE[cls_idx]
            border = CLASS_BORDER[cls_idx]
            # Leaf: show predicted class + how many samples ended here
            total  = int(tree_.n_node_samples[0])
            pct    = n_samples / total * 100
            label  = f"{class_names[cls_idx]}\n{n_samples} samples ({pct:.0f}%)"
            node_w, node_h = 2.8, 0.90
        else:
            face   = INT_FACE
            border = INT_BORDER
            feat   = tree_.feature[node]
            thr    = tree_.threshold[node]
            label  = (f"{decode_feature(feat)}\n"
                      f"Threshold ≤ {thr:.2f}\n"
                      f"n = {n_samples}")
            node_w, node_h = 3.6, 1.05

        ellipse = Ellipse((x, y), width=node_w, height=node_h,
                          facecolor=face, edgecolor=border,
                          linewidth=2.2, zorder=4)
        ax.add_patch(ellipse)
        ax.text(x, y, label,
                fontsize=8.5, ha='center', va='center',
                fontweight='bold', color='#1a1a1a', zorder=5,
                linespacing=1.5)

        # ── Recurse ───────────────────────────────────
        if not is_leaf:
            l, r = tree_.children_left[node], tree_.children_right[node]
            draw(l, depth + 1, node, is_left=True)
            draw(r, depth + 1, node, is_left=False)

    draw(0, 0)

    xs = [p[0] for p in node_pos.values()]
    ys = [p[1] for p in node_pos.values()]
    ax.set_xlim(min(xs) - 2.5, max(xs) + 2.5)
    ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)

    # ── Legend ────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=INT_FACE,   edgecolor=INT_BORDER,
                       linewidth=1.5, label='Decision Node  (asks a question)'),
    ] + [
        mpatches.Patch(facecolor=CLASS_FACE[i], edgecolor=CLASS_BORDER[i],
                       linewidth=1.5, label=f'Leaf → {class_names[i]}')
        for i in range(len(class_names))
    ]
    ax.legend(handles=legend_items, fontsize=10, loc='lower right',
              title='Node types', title_fontsize=11,
              framealpha=0.95, edgecolor='#cccccc')

    ax.set_title(
        f'Decision Tree Classifier  —  Top {max_depth} Levels   '
        f'(full tree depth: {clf.get_depth()})',
        fontsize=15, fontweight='bold', pad=16
    )
    return fig


fig = plot_oval_tree(dt, CLASSES, max_depth=3, figsize=(26, 12))
plt.tight_layout()
plt.savefig('decision_tree_viz.png', dpi=150, bbox_inches='tight')
plt.show()
print("Decision tree diagram saved: decision_tree_viz.png")

# ── Save model ─────────────────────────────────────────
with open('model_dt.pkl', 'wb') as f:
    pickle.dump(dt, f)
print(f"\n✅ Decision Tree model saved: model_dt.pkl  (accuracy: {acc*100:.2f}%)")
