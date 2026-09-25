from __future__ import annotations
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from vfl.data import build_session_data, DATASET_PRESETS
from vfl.pca_rom import run_pca_rom, project_active_only, one_hot
from vfl.icgm import ICGM
from vfl import evaluation as ev

app = FastAPI(title="VFedRS Studio API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION: dict = {}


# ---------------------------------------------------------------- schemas
class NewSessionRequest(BaseModel):
    preset: str = Field(default="clinical")
    n_samples: int = Field(default=1200, ge=200, le=6000)
    n_passive: int = Field(default=3, ge=1, le=5)
    active_frac: float = Field(default=0.2, ge=0.1, le=0.8)
    aligned_ratio: float = Field(default=0.35, ge=0.02, le=0.9)
    seed: int = Field(default=42)


class AlignRequest(BaseModel):
    aligned_ratio: float = Field(ge=0.02, le=0.9)


class PcaRomRequest(BaseModel):
    n_components: int = Field(default=10, ge=2, le=64)


class IcgmRequest(BaseModel):
    epochs: int = Field(default=100, ge=5, le=300)
    lr_g: float = Field(default=0.015, gt=0, le=1)
    lr_d: float = Field(default=0.01, gt=0, le=1)
    lr_theta: float = Field(default=0.03, gt=0, le=1)
    use_adv: bool = True
    use_gr: bool = True
    disc_every: int = Field(default=3, ge=1, le=20)


def require_session():
    if "data" not in SESSION:
        raise HTTPException(400, "No active session. Create one via POST /api/session/new first.")
    return SESSION


def require_pcarom():
    require_session()
    if "pca_rom" not in SESSION:
        raise HTTPException(400, "Run PCA-ROM federated representation extraction first.")
    return SESSION


# ---------------------------------------------------------------- routes
@app.get("/api/presets")
def list_presets():
    return {k: {"label": v["label"], "n_features": v["n_features"], "n_classes": v["n_classes"], "description": v["description"]} for k, v in DATASET_PRESETS.items()}


@app.post("/api/session/new")
def new_session(req: NewSessionRequest):
    if req.preset not in DATASET_PRESETS:
        raise HTTPException(400, f"Unknown preset '{req.preset}'")

    d = build_session_data(req.preset, req.n_samples, req.active_frac, req.n_passive, req.aligned_ratio, req.seed)

    SESSION.clear()
    SESSION["created_at"] = time.time()
    SESSION["config"] = req.dict()
    SESSION["data"] = d

    return _session_status()


@app.get("/api/session/status")
def session_status():
    return _session_status()


def _session_status():
    if "data" not in SESSION:
        return {"active": False}
    d = SESSION["data"]
    parties = {k: len(v) for k, v in d["party_cols"].items()}
    return {
        "active": True,
        "config": SESSION.get("config"),
        "dataset_label": d["cfg"]["label"],
        "dataset_description": d["cfg"]["description"],
        "n_train": int(len(d["y_train"])),
        "n_test": int(len(d["y_test"])),
        "n_aligned": int(len(d["aligned_idx"])),
        "n_nonaligned": int(len(d["non_aligned_idx"])),
        "n_classes": int(d["n_classes"]),
        "party_feature_counts": parties,
        "steps_completed": {
            "session": True,
            "pca_rom": "pca_rom" in SESSION,
            "icgm": "icgm" in SESSION,
            "evaluation": "evaluation" in SESSION,
        },
    }


@app.post("/api/psi/align")
def realign(req: AlignRequest):
    """Re-run PSI-style alignment with a new aligned ratio, keeping the same underlying dataset."""
    require_session()
    cfg = SESSION["config"]
    cfg["aligned_ratio"] = req.aligned_ratio
    d = build_session_data(cfg["preset"], cfg["n_samples"], cfg["active_frac"], cfg["n_passive"], req.aligned_ratio, cfg["seed"])
    SESSION["data"] = d
    SESSION.pop("pca_rom", None)
    SESSION.pop("icgm", None)
    SESSION.pop("evaluation", None)
    return _session_status()


@app.get("/api/psi/preview")
def psi_preview():
    require_session()
    d = SESSION["data"]
    n_show = 8
    aligned_preview = [{"sample_id": int(i), "status": "aligned", "parties_with_access": ["A"] + [f"P{k}" for k in range(1, len(d["party_cols"]) )]} for i in d["aligned_idx"][:n_show]]
    nonaligned_preview = [{"sample_id": int(i), "status": "non-aligned", "parties_with_access": ["A"]} for i in d["non_aligned_idx"][:n_show]]
    return {
        "aligned_sample_preview": aligned_preview,
        "nonaligned_sample_preview": nonaligned_preview,
        "aligned_ratio": round(len(d["aligned_idx"]) / len(d["y_train"]), 4),
        "party_feature_counts": {k: len(v) for k, v in d["party_cols"].items()},
    }


@app.post("/api/pca-rom/run")
def run_pca_rom_endpoint(req: PcaRomRequest):
    d = require_session()["data"]
    X_train, y_train = d["X_train"], d["y_train"]
    aligned_idx = d["aligned_idx"]

    party_data = {k: X_train[np.ix_(aligned_idx, cols)] for k, cols in d["party_cols"].items()}
    y_aligned = y_train[aligned_idx]

    result = run_pca_rom(party_data, y_aligned, d["n_classes"], req.n_components)

    SESSION["pca_rom"] = result
    SESSION["pca_rom_r"] = result["n_components"]

    return {
        "explained_variance_ratio": [round(v, 5) for v in result["explained_variance_ratio"]],
        "cumulative_variance": [round(v, 5) for v in np.cumsum(result["explained_variance_ratio"]).tolist()],
        "mask_reconstruction_error": result["mask_reconstruction_error"],
        "n_components": result["n_components"],
        "party_order": result["party_order"],
        "party_dims": result["party_dims"],
        "per_party_components_preview": result["per_party_components_preview"],
        "h_fed_dim": [int(result["H_fed"].shape[0]), int(result["H_fed"].shape[1])],
    }


@app.post("/api/icgm/train")
def train_icgm(req: IcgmRequest):
    session = require_pcarom()
    d = session["data"]
    pca = session["pca_rom"]
    r = pca["n_components"]

    X_train, y_train = d["X_train"], d["y_train"]
    aligned_idx, non_aligned_idx = d["aligned_idx"], d["non_aligned_idx"]
    active_cols = d["party_cols"]["A"]
    passive_dims = sum(len(v) for k, v in d["party_cols"].items() if k != "A")

    XA_full = X_train[:, active_cols]
    XA_mean = XA_full[aligned_idx].mean(axis=0, keepdims=True)
    XA_aligned = XA_full[aligned_idx] - XA_mean
    XA_nonaligned = XA_full[non_aligned_idx] - XA_mean

    H_fed = pca["H_fed"]
    y_aligned = y_train[aligned_idx]
    y_nonaligned = y_train[non_aligned_idx]
    y_aligned_oh = one_hot(y_aligned, d["n_classes"])
    y_nonaligned_oh = one_hot(y_nonaligned, d["n_classes"]) if len(y_nonaligned) else None

    icgm = ICGM(active_dim=XA_aligned.shape[1], noise_dim=passive_dims, rep_dim=r, n_classes=d["n_classes"])
    rng = np.random.default_rng(5)
    icgm.pretrain_task_head(H_fed, y_aligned_oh, epochs=60, lr=0.05, rng=rng)

    history = icgm.train(
        XA_aligned, XA_nonaligned, H_fed, y_aligned_oh, y_nonaligned_oh,
        epochs=req.epochs, lr_g=req.lr_g, lr_d=req.lr_d, lr_theta=req.lr_theta,
        use_adv=req.use_adv, use_gr=req.use_gr, disc_every=req.disc_every,
    )

    SESSION["icgm"] = icgm
    SESSION["icgm_XA_mean"] = XA_mean
    SESSION["icgm_history"] = history
    SESSION.pop("evaluation", None)

    synth_aligned = icgm.synthesize(XA_aligned)
    final_mse = ev.representation_mse(synth_aligned, H_fed)

    return {
        "history": history,
        "final_mse_per_dim": round(final_mse, 6),
        "config": req.dict(),
    }


@app.post("/api/evaluate/run")
def run_evaluation():
    session = require_pcarom()
    if "icgm" not in session:
        raise HTTPException(400, "Train ICGM representation synthesis first.")

    d = session["data"]
    pca = session["pca_rom"]
    icgm: ICGM = session["icgm"]
    XA_mean = session["icgm_XA_mean"]

    aligned_idx, non_aligned_idx = d["aligned_idx"], d["non_aligned_idx"]
    active_cols = d["party_cols"]["A"]
    X_train, y_train = d["X_train"], d["y_train"]
    X_test, y_test = d["X_test"], d["y_test"]

    y_aligned = y_train[aligned_idx]
    y_nonaligned = y_train[non_aligned_idx]

    XA_full_train = X_train[:, active_cols]
    XA_test = X_test[:, active_cols] - XA_mean

    # ---- LOCAL: raw active-party features only ----
    acc_local = ev.evaluate_local(XA_full_train, y_train, X_test[:, active_cols], y_test)

    # ---- Vanilla VFL: aligned only, passive contribution zeroed at test time ----
    V_A = pca["V_blocks"]["A"]
    H_fed_active_only = (XA_full_train[aligned_idx] - XA_mean) @ V_A
    XA_test_proj = XA_test @ V_A
    acc_vfl = ev.evaluate_vfl_vanilla(H_fed_active_only, y_aligned, XA_test_proj, y_test)

    # ---- PCA-ROM variant (ablation): true H_fed for aligned, active-only projection for non-aligned ----
    XA_nonaligned_proj = project_active_only(XA_full_train[non_aligned_idx] - XA_mean, V_A)
    acc_pcarom = ev.evaluate_pca_rom_variant(pca["H_fed"], XA_nonaligned_proj, y_aligned, y_nonaligned, XA_test_proj, y_test)

    # ---- Full VFedRS: ICGM-synthesized representations everywhere ----
    XA_aligned_c = XA_full_train[aligned_idx] - XA_mean
    XA_nonaligned_c = XA_full_train[non_aligned_idx] - XA_mean
    synth_aligned = icgm.synthesize(XA_aligned_c, seed=101)
    synth_nonaligned = icgm.synthesize(XA_nonaligned_c, seed=102) if len(XA_nonaligned_c) else np.empty((0, pca["n_components"]))
    synth_test = icgm.synthesize(XA_test, seed=103)
    acc_vfedrs = ev.evaluate_vfedrs(synth_aligned, synth_nonaligned, y_aligned, y_nonaligned, synth_test, y_test)

    mse_aligned = ev.representation_mse(synth_aligned, pca["H_fed"])

    results = {
        "LOCAL": round(acc_local, 2),
        "VFL (vanilla)": round(acc_vfl, 2),
        "PCA-ROM only": round(acc_pcarom, 2),
        "VFedRS (full)": round(acc_vfedrs, 2),
    }
    SESSION["evaluation"] = results

    comm_cost = _communication_costs(d)

    return {
        "accuracy": results,
        "representation_mse_aligned": round(mse_aligned, 6),
        "n_aligned": int(len(aligned_idx)),
        "n_nonaligned": int(len(non_aligned_idx)),
        "communication_cost_estimate": comm_cost,
    }


def _communication_costs(d):
    n_aligned = len(d["aligned_idx"])
    F_a = sum(len(v) for v in d["party_cols"].values())
    K = len(d["party_cols"]) - 1
    r = SESSION.get("pca_rom", {}).get("n_components", 16)

    vfedrs_cost = n_aligned ** 2 + F_a ** 2 + F_a * n_aligned
    T_iters = 150
    vfl_cost = T_iters * K * r * n_aligned

    return {
        "vfedrs_relative_units": int(vfedrs_cost),
        "vanilla_vfl_relative_units": int(vfl_cost),
        "speedup_factor": round(vfl_cost / max(vfedrs_cost, 1), 1),
        "note": "Relative units following the paper's O(|Ia|^2 + |Fa|^2 + |Fa||Ia|) vs O(T*K*r*|Ia|) analysis. VFedRS communicates in 4 fixed rounds; vanilla VFL communicates every training iteration (T~150).",
    }


@app.get("/api/privacy/report")
def privacy_report():
    session = require_session()
    d = session["data"]
    pca = session.get("pca_rom")
    return {
        "mask_reconstruction_error": pca["mask_reconstruction_error"] if pca else None,
        "assumptions": [
            {"entity": "Authority", "assumption": "Semi-honest; generates and distributes masks Q, R then goes offline. Receives no data from any party."},
            {"entity": "Active Party (A)", "assumption": "Semi-honest; holds labels and its own features. Cannot infer passive parties' raw features or local SVD blocks."},
            {"entity": "Passive Parties (P1..PK)", "assumption": "Semi-honest; hold only feature subsets. Cannot access raw data, labels, or masked representations of the active party."},
        ],
        "protocol_rounds": [
            "Round 1 — Authority distributes orthogonal masks Q (sample-dim) and R (feature-dim, split per party).",
            "Round 2 — Each passive party sends masked local data and masked mask-fragments to the active party.",
            "Round 3 — Active party returns masked principal components to each passive party.",
            "Round 4 — Each passive party contributes its locally-transformed (never raw) features via secure aggregation.",
        ],
        "what_leaves_each_party": {
            "Active Party": "Only masked matrices (Q·X_A·R_A) and masked labels (Q·Y) — never raw features or labels.",
            "Passive Parties": "Only masked matrices (Q·X_k·R_k) and mask fragments — never raw features.",
        },
        "party_feature_counts": {k: len(v) for k, v in d["party_cols"].items()},
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}
