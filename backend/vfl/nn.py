"""
Minimal numpy-only neural network toolkit.

We avoid heavy DL dependencies (torch) so the backend installs fast and
stays dependency-light, while still doing *real* forward/backward passes
and gradient updates (Adam) rather than faking numbers.
"""
from __future__ import annotations
import numpy as np


class Dense:
    """A single fully-connected layer with optional activation and its own Adam state."""

    def __init__(self, in_dim: int, out_dim: int, activation: str | None = None, seed: int | None = None):
        rng = np.random.default_rng(seed)
        limit = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, limit, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.activation = activation

        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t = 0

        self._x = None
        self._z = None
        self._out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._z = x @ self.W + self.b
        if self.activation == "relu":
            self._out = np.maximum(0.0, self._z)
        elif self.activation == "sigmoid":
            self._out = 1.0 / (1.0 + np.exp(-np.clip(self._z, -30, 30)))
        elif self.activation == "leaky_relu":
            self._out = np.where(self._z > 0, self._z, 0.01 * self._z)
        else:
            self._out = self._z
        return self._out

    def backward(self, dout: np.ndarray, lr: float = 1e-3) -> np.ndarray:
        if self.activation == "relu":
            dz = dout * (self._z > 0)
        elif self.activation == "leaky_relu":
            dz = dout * np.where(self._z > 0, 1.0, 0.01)
        elif self.activation == "sigmoid":
            dz = dout * self._out * (1 - self._out)
        else:
            dz = dout

        n = max(len(self._x), 1)
        dW = self._x.T @ dz / n
        db = dz.mean(axis=0)
        dx = dz @ self.W.T

        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        self.mW = beta1 * self.mW + (1 - beta1) * dW
        self.vW = beta2 * self.vW + (1 - beta2) * (dW ** 2)
        mW_hat = self.mW / (1 - beta1 ** self.t)
        vW_hat = self.vW / (1 - beta2 ** self.t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

        self.mb = beta1 * self.mb + (1 - beta1) * db
        self.vb = beta2 * self.vb + (1 - beta2) * (db ** 2)
        mb_hat = self.mb / (1 - beta1 ** self.t)
        vb_hat = self.vb / (1 - beta2 ** self.t)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

        return dx


class MLP:
    """A tiny 2-layer MLP composed of Dense layers, with a manual backward chain."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, out_activation: str | None = None, seed: int | None = None):
        self.l1 = Dense(in_dim, hidden_dim, activation="leaky_relu", seed=seed)
        self.l2 = Dense(hidden_dim, out_dim, activation=out_activation, seed=None if seed is None else seed + 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.l1.forward(x)
        return self.l2.forward(h)

    def backward(self, dout: np.ndarray, lr: float = 1e-3) -> np.ndarray:
        dh = self.l2.backward(dout, lr)
        return self.l1.backward(dh, lr)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_grad(probs: np.ndarray, onehot: np.ndarray) -> tuple[float, np.ndarray]:
    n = probs.shape[0]
    loss = -np.mean(np.sum(onehot * np.log(np.clip(probs, 1e-9, 1.0)), axis=1))
    grad = (probs - onehot) / n
    return float(loss), grad


def bce_grad(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    pred_c = np.clip(pred, 1e-7, 1 - 1e-7)
    loss = -np.mean(target * np.log(pred_c) + (1 - target) * np.log(1 - pred_c))
    grad = (pred_c - target) / (pred_c * (1 - pred_c) * len(pred)) * (pred_c * (1 - pred_c))
    # simplifies to (pred - target) / n but kept explicit for clarity of BCE+sigmoid combo upstream
    grad = (pred_c - target) / len(pred)
    return float(loss), grad
