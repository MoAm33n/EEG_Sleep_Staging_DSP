import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC   # Şimdilik kapattım, istersen geri açarsın

from scipy.stats import randint, uniform
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG =================
DATA_DIR = '/home/mohamed/Uni Stuff/Signals and Systems Project/EEG_Sleep_Staging_DSP/data'
FEATURE_PATH = os.path.join(DATA_DIR, 'feature_matrix.csv')
MODEL_SAVE_PATH = os.path.join(DATA_DIR, 'best_sleep_model_subjectwise.joblib')

BINARY_CLASSIFICATION = False   # True => Wake vs Sleep
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Aramayı hafifletmek için:
FAST_MODE = False    # True yaparsan daha hızlı, False yaparsan daha agresif arar

if FAST_MODE:
    N_ITER_SEARCH = 10
    N_SPLITS_CV = 3
else:
    N_ITER_SEARCH = 25
    N_SPLITS_CV = 5

# ============== LOAD FEATURES ==========
print(f"Loading feature matrix from: {FEATURE_PATH}")
df = pd.read_csv(FEATURE_PATH)

if 'target' not in df.columns or 'subject_id' not in df.columns:
    raise ValueError("CSV must contain 'target' and 'subject_id' columns.")

# -1 label (M, ?) temizle
before = len(df)
df = df[df['target'] != -1]
after = len(df)
print(f"Removed {before - after} epochs with label -1. Remaining: {after}")

subject_ids = df['subject_id'].values
y = df['target'].values
X = df.drop(columns=['target', 'subject_id']).values

print("Class distribution (original):", np.unique(y, return_counts=True))

# ======= OPTIONAL BINARY TRANSFORM =====
if BINARY_CLASSIFICATION:
    # Wake (1) -> 0, diğerleri -> 1
    y = np.where(y == 1, 0, 1)
    print("Using BINARY classification: 0=Wake, 1=Sleep")
    print("Binary class distribution:", np.unique(y, return_counts=True))
else:
    print("Using MULTI-CLASS classification with original sleep stages.")

# ======= SUBJECT-WISE TRAIN/TEST SPLIT =====
unique_subjects = np.unique(subject_ids)
train_subj, test_subj = train_test_split(
    unique_subjects,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_mask = np.isin(subject_ids, train_subj)
test_mask = np.isin(subject_ids, test_subj)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
groups_train = subject_ids[train_mask]

print("Train subjects:", len(np.unique(train_subj)), "Test subjects:", len(np.unique(test_subj)))
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ======= MODELS & PARAM GRIDS ==========
# İstersen sadece RF bırak, LogReg'i kapat; ya da tam tersi.
models_and_params = {
    "log_reg": {
        "model": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "params": {
            "model__C": uniform(0.01, 5.0),
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"]
        }
    },
    "random_forest": {
        "model": RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "params": {
            # HAFİFLETİLMİŞ ARALIKLAR
            "model__n_estimators": randint(100, 250),
            "model__max_depth": randint(5, 20),
            "model__min_samples_split": randint(2, 10),
            "model__min_samples_leaf": randint(1, 5),
            "model__max_features": ["sqrt", "log2"]
        }
    },
    # "svm": {
    #     "model": SVC(
    #         class_weight="balanced"
    #     ),
    #     "params": {
    #         "model__C": uniform(0.1, 10.0),
    #         "model__gamma": ["scale", "auto"],
    #         "model__kernel": ["rbf"]
    #     }
    # }
}

group_kfold = GroupKFold(n_splits=N_SPLITS_CV)

best_model = None
best_model_name = None
best_val_score = -np.inf

# ======= HYPERPARAM SEARCH (SUBJECT-WISE CV) =====
for name, mp in models_and_params.items():
    print(f"\n### Trying model: {name} ###")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", mp["model"])
    ])

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=mp["params"],
        n_iter=N_ITER_SEARCH,
        scoring="accuracy",
        cv=group_kfold,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2  # daha çok log gör
    )

    search.fit(X_train, y_train, **{"groups": groups_train})

    print(f"  -> Best CV accuracy for {name}: {search.best_score_:.4f}")
    print(f"  -> Best params: {search.best_params_}")

    if search.best_score_ > best_val_score:
        best_val_score = search.best_score_
        best_model = search.best_estimator_
        best_model_name = name

print("\n=========================================")
print(f"Best model: {best_model_name}")
print(f"Best CV accuracy (subject-wise): {best_val_score:.4f}")
print("=========================================")

# ======= TEST SET PERFORMANCE ===========
y_pred = best_model.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy (subject-wise split): {test_acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

stage_labels = ["W", "N1", "N2", "N3", "R"]
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels= stage_labels, yticklabels=stage_labels)
plt.title('Confusion Matrix (Subject-wise Test)')
plt.xlabel('Predicted_Stage')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# ======= SAVE BEST MODEL ================
joblib.dump(best_model, MODEL_SAVE_PATH)
print(f"\nSaved best model to: {MODEL_SAVE_PATH}")