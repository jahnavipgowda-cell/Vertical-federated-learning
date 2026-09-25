from __future__ import annotations
import numpy as np
from .nn import MLP, Dense, softmax, cross_entropy_grad


class ICGM:
    """
    Input-Conditional Generative Model.

    Generator:      G([x_A, z]) -> synthetic representation h  (matches H_fed's dimensionality)
    Discriminator:  D(h) -> real/synthetic probability (trained only on aligned samples)
    Task head:      theta(h) -> class logits, pretrained on H_fed, then used to backprop a
                     generative-regression signal into G using BOTH aligned and non-aligned samples
                     (Algorithm 2 in the paper).
    """

    def __init__(self, active_dim: int, noise_dim: int, rep_dim: int, n_classes: int, hidden: int = 48, seed: int = 3):
        self.rep_dim = rep_dim
        self.noise_dim = noise_dim
        self.n_classes = n_classes
        self.G = MLP(active_dim + noise_dim, hidden, rep_dim, out_activation=None, seed=seed)
        self.D = MLP(rep_dim, hidden, 1, out_activation="sigmoid", seed=seed + 100)
        self.theta = Dense(rep_dim, n_classes, activation=None, seed=seed + 200)

    def _noise(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0, 1, size=(n, self.noise_dim))

    def pretrain_task_head(self, H_fed: np.ndarray, y_aligned_onehot: np.ndarray, epochs: int, lr: float, rng: np.random.Generator):
        history = []
        n = len(H_fed)
        for _ in range(epochs):
            perm = rng.permutation(n)
            logits = self.theta.forward(H_fed[perm])
            probs = softmax(logits)
            loss, grad = cross_entropy_grad(probs, y_aligned_onehot[perm])
            self.theta.backward(grad, lr)
            history.append(loss)
        return history

    def train(
        self,
        X_active_aligned: np.ndarray,
        X_active_nonaligned: np.ndarray,
        H_fed: np.ndarray,
        y_aligned_onehot: np.ndarray,
        y_nonaligned_onehot: np.ndarray | None,
        epochs: int,
        lr_g: float,
        lr_d: float,
        lr_theta: float,
        use_adv: bool,
        use_gr: bool,
        disc_every: int,
        seed: int = 11,
        reg_weight: float = 1e-3,
    ):
        rng = np.random.default_rng(seed)
        history = {"l_adv": [], "l_disc": [], "l_gr": [], "mse_aligned": [], "epoch": []}

        has_nonaligned = X_active_nonaligned is not None and len(X_active_nonaligned) > 0

        for epoch in range(epochs):
            # ---- aligned batch: adversarial + generative-regression ----
            z_a = self._noise(len(X_active_aligned), rng)
            g_in_a = np.hstack([X_active_aligned, z_a])
            h_synth_a = self.G.forward(g_in_a)

            l_adv_val = 0.0
            l_disc_val = 0.0
            if use_adv:
                # Discriminator forward on real + fake
                d_real = self.D.forward(H_fed)
                d_fake = self.D.forward(h_synth_a)

                # Generator wants D(fake) -> 1  (non-saturating GAN loss)
                eps = 1e-7
                l_adv_val = float(-np.mean(np.log(np.clip(d_fake, eps, 1.0))))
                dL_dfake = -1.0 / (np.clip(d_fake, eps, 1.0) * len(d_fake))

                # Backprop through D to get gradient wrt its input (h_synth_a), WITHOUT updating D here
                d_hidden_grad = dL_dfake * (d_fake * (1 - d_fake))  # sigmoid derivative
                dx_from_top = d_hidden_grad @ self.D.l2.W.T
                dz_hidden = dx_from_top * (self.D.l1._z > 0).astype(float) + dx_from_top * (self.D.l1._z <= 0) * 0.01
                dh_synth_from_adv = dz_hidden @ self.D.l1.W.T

            else:
                dh_synth_from_adv = np.zeros_like(h_synth_a)

            mse_aligned = float(np.mean((h_synth_a - H_fed) ** 2))
            dh_from_mse = 2.0 * (h_synth_a - H_fed) / h_synth_a.size

            adv_weight = 1.0 if use_adv else 0.0
            total_dh_a = adv_weight * dh_synth_from_adv * 0.05 + dh_from_mse

            self.G.backward(total_dh_a, lr_g)

            if use_adv and (epoch % disc_every == 0):
                d_real = self.D.forward(H_fed)
                d_fake = self.D.forward(self.G.forward(g_in_a))
                eps = 1e-7
                l_disc_val = float(-np.mean(np.log(np.clip(d_real, eps, 1.0))) - np.mean(np.log(np.clip(1 - d_fake, eps, 1.0))))
                grad_real = -1.0 / (np.clip(d_real, eps, 1.0) * len(d_real)) * (d_real * (1 - d_real))
                self.D.backward(grad_real, lr_d)
                d_fake2 = self.D.forward(h_synth_a)
                grad_fake = 1.0 / (np.clip(1 - d_fake2, eps, 1.0) * len(d_fake2)) * (d_fake2 * (1 - d_fake2))
                self.D.backward(grad_fake, lr_d)

            # ---- generative regression loss on ALL samples (aligned + non-aligned) ----
            l_gr_val = 0.0
            if use_gr:
                if has_nonaligned:
                    X_all = np.vstack([X_active_aligned, X_active_nonaligned])
                    y_all = np.vstack([y_aligned_onehot, y_nonaligned_onehot]) if y_nonaligned_onehot is not None else y_aligned_onehot
                else:
                    X_all = X_active_aligned
                    y_all = y_aligned_onehot

                z_all = self._noise(len(X_all), rng)
                g_in_all = np.hstack([X_all, z_all])
                h_synth_all = self.G.forward(g_in_all)

                logits = self.theta.forward(h_synth_all)
                probs = softmax(logits)
                l_gr_val, grad_logits = cross_entropy_grad(probs, y_all)

                # variance regularizer Omega: discourage collapsed / exploding representations
                var_penalty_grad = 2 * reg_weight * (h_synth_all - h_synth_all.mean(axis=0, keepdims=True)) / h_synth_all.size

                dz = grad_logits @ self.theta.W.T
                self.G.backward(dz + var_penalty_grad, lr_g)

                # occasionally refine task head using synthetic non-aligned reps + labels (Algorithm 2)
                if has_nonaligned and (epoch % disc_every == 0):
                    z_na = self._noise(len(X_active_nonaligned), rng)
                    h_na = self.G.forward(np.hstack([X_active_nonaligned, z_na]))
                    logits_na = self.theta.forward(h_na)
                    probs_na = softmax(logits_na)
                    if y_nonaligned_onehot is not None:
                        _, grad_na = cross_entropy_grad(probs_na, y_nonaligned_onehot)
                        self.theta.backward(grad_na, lr_theta)

            history["l_adv"].append(round(l_adv_val, 5))
            history["l_disc"].append(round(l_disc_val, 5))
            history["l_gr"].append(round(l_gr_val, 5))
            history["mse_aligned"].append(round(mse_aligned, 5))
            history["epoch"].append(epoch + 1)

        return history

    def synthesize(self, X_active: np.ndarray, seed: int = 99) -> np.ndarray:
        rng = np.random.default_rng(seed)
        z = self._noise(len(X_active), rng)
        return self.G.forward(np.hstack([X_active, z]))
