from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def _fit_predict(X_train, y_train, X_test, y_test, seed=0):
    clf = RandomForestClassifier(n_estimators=80, max_depth=9, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return float(accuracy_score(y_test, preds)) * 100.0


def evaluate_local(X_active_train_all, y_train_all, X_active_test, y_test):
    return _fit_predict(X_active_train_all, y_train_all, X_active_test, y_test)


def evaluate_vfl_vanilla(H_fed_aligned_active_only, y_aligned, X_active_test_proj, y_test):
    """Vanilla VFL: only trained on aligned samples; at test time the passive parties are
    unavailable so their contribution is zeroed out (as in the paper's evaluation protocol)."""
    return _fit_predict(H_fed_aligned_active_only, y_aligned, X_active_test_proj, y_test)


def evaluate_pca_rom_variant(H_fed_aligned, active_only_nonaligned_rep, y_aligned, y_nonaligned, X_active_test_proj, y_test):
    """Ablation: PCA-ROM only (no ICGM) — non-aligned samples get just the active party's own
    projection, with no knowledge transferred from passive parties."""
    if len(active_only_nonaligned_rep) > 0:
        X_train = np.vstack([H_fed_aligned, active_only_nonaligned_rep])
        y_train = np.concatenate([y_aligned, y_nonaligned])
    else:
        X_train, y_train = H_fed_aligned, y_aligned
    return _fit_predict(X_train, y_train, X_active_test_proj, y_test)


def evaluate_vfedrs(synth_aligned, synth_nonaligned, y_aligned, y_nonaligned, synth_test, y_test):
    """Full VFedRS: ICGM-synthesized representations for ALL local samples (aligned + non-aligned)."""
    if len(synth_nonaligned) > 0:
        X_train = np.vstack([synth_aligned, synth_nonaligned])
        y_train = np.concatenate([y_aligned, y_nonaligned])
    else:
        X_train, y_train = synth_aligned, y_aligned
    return _fit_predict(X_train, y_train, synth_test, y_test)


def representation_mse(synth: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((synth - target) ** 2))
