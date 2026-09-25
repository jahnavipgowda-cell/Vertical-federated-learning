from __future__ import annotations
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_PRESETS = {
    "clinical": {
        "label": "Clinical Risk (MIMIC-III style)",
        "n_features": 24,
        "n_classes": 4,
        "n_informative": 16,
        "description": "Synthetic patient records modeled on the length-of-stay prediction task (MIMIC-III) used in the paper.",
    },
    "activity": {
        "label": "Sensor Activity (HAPT style)",
        "n_features": 40,
        "n_classes": 6,
        "n_informative": 28,
        "description": "Synthetic smartphone sensor readings modeled on the human activity recognition task (HAPT).",
    },
    "vision": {
        "label": "Pixel-Block Vision (MNIST style)",
        "n_features": 32,
        "n_classes": 10,
        "n_informative": 24,
        "description": "Synthetic flattened pixel-block features modeled on vertically-split image classification (MNIST/FMNIST).",
    },
    "finance": {
        "label": "Cross-Institution Finance",
        "n_features": 28,
        "n_classes": 2,
        "n_informative": 18,
        "description": "Synthetic bank + e-commerce + insurer style features for a credit-risk style task.",
    },
}


def generate_dataset(preset: str, n_samples: int, seed: int = 42):
    cfg = DATASET_PRESETS[preset]
    X, y = make_classification(
        n_samples=n_samples,
        n_features=cfg["n_features"],
        n_informative=cfg["n_informative"],
        n_redundant=max(0, cfg["n_features"] - cfg["n_informative"] - 2),
        n_classes=cfg["n_classes"],
        n_clusters_per_class=1,
        class_sep=1.6,
        random_state=seed,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, cfg


def partition_features(X: np.ndarray, active_frac: float, n_passive: int, seed: int = 42):
    """Vertically partition feature columns among the active party and K passive parties."""
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    perm = rng.permutation(n_features)

    n_active = max(2, int(round(n_features * active_frac)))
    n_active = min(n_active, n_features - n_passive)  # leave >=1 feature per passive party
    active_cols = perm[:n_active]
    remaining = perm[n_active:]
    passive_splits = np.array_split(remaining, n_passive)

    parties = {"A": active_cols.tolist()}
    for k, cols in enumerate(passive_splits, start=1):
        parties[f"P{k}"] = cols.tolist()
    return parties


def split_aligned(n_samples: int, aligned_ratio: float, seed: int = 42):
    """Simulate PSI: which sample ids are 'aligned' (visible to all parties) vs only active."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    n_aligned = max(4, int(round(n_samples * aligned_ratio)))
    aligned_idx = np.sort(idx[:n_aligned])
    non_aligned_idx = np.sort(idx[n_aligned:])
    return aligned_idx, non_aligned_idx


def build_session_data(preset: str, n_samples: int, active_frac: float, n_passive: int, aligned_ratio: float, seed: int = 42):
    X, y, cfg = generate_dataset(preset, n_samples, seed)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.25, random_state=seed, stratify=y
    )

    party_cols = partition_features(X_train, active_frac, n_passive, seed)
    aligned_idx, non_aligned_idx = split_aligned(len(y_train), aligned_ratio, seed)

    return {
        "cfg": cfg,
        "preset": preset,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "party_cols": party_cols,
        "aligned_idx": aligned_idx,
        "non_aligned_idx": non_aligned_idx,
        "n_classes": cfg["n_classes"],
    }
