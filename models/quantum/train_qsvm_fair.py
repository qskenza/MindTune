# models/quantum/train_qsvm_fair.py

from __future__ import annotations
import glob
import os
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from preprocessing.feature_builder import build_window_features

RAW_DATA_DIR = "data/hr_sessions"
WINDOW_SIZE = 20
STEP = 10

# Use only a few strong features for QSVM
FEATURES = [
    "hr_mean",
    "hr_std",
    "rmssd",
    "hr_range"
]

# If face features exist later, extend like this:
# FEATURES = ["hr_mean", "hr_std", "stress_conf_mean", "neutral_conf_mean"]


def load_all_sessions(data_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def balance_training_set(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    df = X_train.copy()
    df["label"] = y_train.values

    counts = df["label"].value_counts()
    min_count = counts.min()

    parts = []
    for cls in counts.index:
        cls_df = df[df["label"] == cls]
        cls_sample = resample(cls_df, replace=False, n_samples=min_count, random_state=42)
        parts.append(cls_sample)

    out = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    Xb = out.drop(columns=["label"]).values
    yb = out["label"].values
    return Xb, yb


def evaluate(name: str, y_true, y_pred, train_time=None, test_time=None):
    print(f"\n{name}")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred, zero_division=0), 4))
    print("Recall   :", round(recall_score(y_true, y_pred, zero_division=0), 4))
    print("F1-score :", round(f1_score(y_true, y_pred, zero_division=0), 4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    if train_time is not None:
        print("Train time:", round(train_time, 4), "sec")
    if test_time is not None:
        print("Test time :", round(test_time, 4), "sec")


def main():
    raw_df = load_all_sessions(RAW_DATA_DIR)

    feature_df = build_window_features(
        raw_df,
        window_size=WINDOW_SIZE,
        step=STEP,
    )

    needed_cols = FEATURES + ["label", "session_id"]
    feature_df = feature_df.dropna(subset=needed_cols).reset_index(drop=True)

    X = feature_df[FEATURES].copy()
    y = feature_df["label"].copy()
    groups = feature_df["session_id"].copy()

    print("Feature dataframe shape:", feature_df.shape)
    print("Class counts:\n", y.value_counts())

    gkf = GroupKFold(n_splits=min(5, len(groups.unique())))

    rf_scores = []
    qsvm_scores = []

    fold_num = 1
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        print(f"\n========== Fold {fold_num} ==========")

        X_train_df = X.iloc[train_idx].copy()
        X_test_df = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        # Balance ONLY the training data
        X_train_bal, y_train_bal = balance_training_set(X_train_df, y_train)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_bal)
        X_test_scaled = scaler.transform(X_test_df.values)

        # Optional PCA to help QSVM
        n_components = min(3, X_train_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)

        # ---------- Random Forest ----------
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight=None,
        )

        t0 = time.time()
        rf.fit(X_train_final, y_train_bal)
        rf_train_time = time.time() - t0

        t0 = time.time()
        rf_pred = rf.predict(X_test_final)
        rf_test_time = time.time() - t0

        evaluate("Random Forest", y_test, rf_pred, rf_train_time, rf_test_time)
        rf_scores.append(f1_score(y_test, rf_pred, zero_division=0))

        # ---------- QSVM ----------
        feature_map = ZZFeatureMap(feature_dimension=X_train_final.shape[1], reps=1)
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
        qsvm = QSVC(quantum_kernel=quantum_kernel)

        t0 = time.time()
        qsvm.fit(X_train_final, y_train_bal)
        qsvm_train_time = time.time() - t0

        t0 = time.time()
        qsvm_pred = qsvm.predict(X_test_final)
        qsvm_test_time = time.time() - t0

        evaluate("QSVM", y_test, qsvm_pred, qsvm_train_time, qsvm_test_time)
        qsvm_scores.append(f1_score(y_test, qsvm_pred, zero_division=0))

        fold_num += 1

    print("\n==============================")
    print("Average RF F1   :", round(float(np.mean(rf_scores)), 4))
    print("Average QSVM F1 :", round(float(np.mean(qsvm_scores)), 4))
    print("==============================")


if __name__ == "__main__":
    main()