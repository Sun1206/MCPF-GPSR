"""Extended MCPF-Cal validation for saved BINTS predictions.

This script is a post-hoc calibrator: it never retrains BINTS.  It reads saved
validation/test predictions, constructs temporal and graph-filtered residual
directions, fits small validation-only heads, and writes comparable test rows.

Methods:
  BINTS                     frozen parent
  AffineH                   horizon-wise affine calibration control
  PRC                       periodic residual correction
  GAR                       graph-anchor residual, W(A-P)
  GCR                       graph-contrast residual, (I-W)(A-P)
  MCPF-AO-Route             ancestor-orthogonal surplus-routed residual path
  MCPF-GPSR                 graph-periodic surplus readout over structured forecasts
  MCPF-MetricAware-MSE/MAE  metric-specific GPSR/PRC deployment policies
  MCPF-Select-MSE/MAE       validation-selected deployed heads for a target metric
  GenericResidualRidge      ridge over residual directions [D_T,D_G,D_C]
  GenericMLPReadout         generic nonlinear stacking over the GPSR forecast basis
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - the baseline is skipped if sklearn is unavailable.
    MLPRegressor = None
    make_pipeline = None
    StandardScaler = None


DATASET_NODES = {
    "busan": 103,
    "daegu": 85,
    "seoul": 233,
    "covid": 16,
    "nyc": 10,
    "nyc_covid": 5,
    "busan_new": 60,
    "daegu_new": 61,
    "seoul_new": 128,
}

ADJ_FILES = {
    "busan": "busan_adj_matrix_with_diag_1.npy",
    "daegu": "daegu_adj_matrix_with_diag_1.npy",
    "seoul": "seoul_adj_matrix_with_diag_1.npy",
    "covid": "nationwide_adj_matrix_with_diag_1.npy",
    "nyc": "nyc_taxi_matrix_with_diag_1.npy",
    "nyc_covid": "nyc_covid_matrix_with_diag_1.npy",
    "busan_new": "busan_adj_matrix_with_diag_1_60.npy",
    "daegu_new": "daegu_adj_matrix_with_diag_1_61.npy",
    "seoul_new": "seoul_adj_matrix_with_diag_1_128.npy",
}


@dataclass
class HeadResult:
    method: str
    val_pred: np.ndarray
    test_pred: np.ndarray
    detail: dict[str, object]


def mse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred.astype(np.float64) - y_true.astype(np.float64)
    return float(np.mean(err * err)), float(np.mean(np.abs(err)))


def periodic_anchor(x: np.ndarray, pred_len: int, cycle: int) -> np.ndarray:
    n, seq_len, dim = x.shape
    if cycle <= 0 or cycle > seq_len:
        return np.repeat(x[:, -1:, :], pred_len, axis=1)
    idx = [seq_len - cycle + (h % cycle) for h in range(pred_len)]
    return x[:, idx, :]


def row_normalize(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    row_sum = w.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 1e-12] = 1.0
    return w / row_sum


def _load_adjacency_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        loaded = np.load(path)
        first_key = loaded.files[0]
        return loaded[first_key]
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            try:
                obj = pickle.load(f)
            except UnicodeDecodeError:
                f.seek(0)
                obj = pickle.load(f, encoding="latin1")
        if isinstance(obj, np.ndarray):
            return obj
        if isinstance(obj, dict):
            for key in ("adj_mx", "adj", "matrix", "A"):
                if key in obj:
                    return np.asarray(obj[key])
        if isinstance(obj, (tuple, list)):
            for item in reversed(obj):
                arr = np.asarray(item)
                if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                    return arr
    raise ValueError(f"Unsupported adjacency file: {path}")


def load_adjacency(
    dataset: str,
    root: Path | None,
    dim: int,
    nodes: int | None,
    graph_mode: str = "real",
    graph_seed: int = 0,
    adj_path: Path | None = None,
) -> tuple[np.ndarray, int, str]:
    dataset = dataset.lower()
    n = nodes or DATASET_NODES.get(dataset)
    source = "identity"
    w = None
    if adj_path is not None:
        if not adj_path.exists():
            raise FileNotFoundError(adj_path)
        w = _load_adjacency_file(adj_path)
        source = str(adj_path)
    if root is not None and dataset in ADJ_FILES:
        path = root / "adj_matrix" / ADJ_FILES[dataset]
        if w is None and path.exists():
            w = np.load(path)
            source = str(path)
    if w is not None:
        n = int(w.shape[0])
    if n is None or dim % n != 0:
        n = dim
        w = np.eye(dim, dtype=np.float64)
        source = f"identity_dim_{dim}"
    elif w is None:
        w = np.eye(n, dtype=np.float64)
        source = f"identity_nodes_{n}"

    if graph_mode == "identity":
        w = np.eye(int(n), dtype=np.float64)
        source = f"identity_nodes_{n}"
    elif graph_mode == "permute":
        rng = np.random.default_rng(graph_seed)
        perm = rng.permutation(int(n))
        w = w[np.ix_(perm, perm)]
        source = f"permuted({source},seed={graph_seed})"
    elif graph_mode == "random":
        rng = np.random.default_rng(graph_seed)
        base = np.asarray(w, dtype=np.float64)
        density = max(float(np.mean(base > 0.0)), 1.0 / max(int(n), 1))
        mask = rng.random((int(n), int(n))) < density
        np.fill_diagonal(mask, True)
        vals = rng.random((int(n), int(n))) * mask
        w = vals
        source = f"random_density_{density:.4f}_seed_{graph_seed}"
    elif graph_mode != "real":
        raise ValueError(f"Unsupported graph_mode: {graph_mode}")
    return row_normalize(w), n, source


def graph_filter(x: np.ndarray, w: np.ndarray, nodes: int) -> np.ndarray:
    """Apply row-normalized W to a flattened graph-indexed tensor."""
    s, h, dim = x.shape
    if dim % nodes != 0:
        return x.copy()
    feat = dim // nodes
    xr = x.reshape(s, h, nodes, feat)
    out = np.einsum("ij,shjf->shif", w, xr, optimize=True)
    return out.reshape(s, h, dim).astype(np.float32)


def fit_affine_horizon(pred_v: np.ndarray, y_v: np.ndarray) -> np.ndarray:
    pred_len = pred_v.shape[1]
    coeff = np.zeros((pred_len, 2), dtype=np.float64)
    for h in range(pred_len):
        x = pred_v[:, h, :].reshape(-1).astype(np.float64)
        y = y_v[:, h, :].reshape(-1).astype(np.float64)
        xm, ym = x.mean(), y.mean()
        xv = np.mean((x - xm) ** 2)
        a = np.mean((x - xm) * (y - ym)) / (xv + 1e-8)
        b = ym - a * xm
        coeff[h] = (a, b)
    return coeff


def apply_affine_horizon(pred: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    out = pred.astype(np.float64).copy()
    for h, (a, b) in enumerate(coeff):
        out[:, h, :] = a * out[:, h, :] + b
    return out.astype(np.float32)


def fit_scalar_direction(
    pred_v: np.ndarray,
    direction_v: np.ndarray,
    y_v: np.ndarray,
    ridge: float,
    shrink: float,
) -> np.ndarray:
    pred_len = pred_v.shape[1]
    lam = np.zeros(pred_len, dtype=np.float64)
    for h in range(pred_len):
        d = direction_v[:, h, :].reshape(-1).astype(np.float64)
        r = (y_v[:, h, :] - pred_v[:, h, :]).reshape(-1).astype(np.float64)
        raw = np.sum(d * r) / (np.sum(d * d) + ridge)
        lam[h] = np.clip(shrink * raw, -1.0, 1.0)
    return lam


def apply_scalar_direction(pred: np.ndarray, direction: np.ndarray, lam: np.ndarray) -> np.ndarray:
    out = pred.astype(np.float64).copy()
    for h, v in enumerate(lam):
        out[:, h, :] = out[:, h, :] + v * direction[:, h, :]
    return out.astype(np.float32)


def fit_best_scalar_head(
    name: str,
    pred_v: np.ndarray,
    pred_t: np.ndarray,
    direction_v: np.ndarray,
    direction_t: np.ndarray,
    y_v: np.ndarray,
    ridges: Iterable[float],
    shrinks: Iterable[float],
) -> HeadResult:
    best = None
    for ridge in ridges:
        for shrink in shrinks:
            lam = fit_scalar_direction(pred_v, direction_v, y_v, ridge, shrink)
            val_hat = apply_scalar_direction(pred_v, direction_v, lam)
            vm, _ = mse_mae(y_v, val_hat)
            if best is None or vm < best[0]:
                best = (vm, ridge, shrink, lam, val_hat)
    assert best is not None
    _, ridge, shrink, lam, val_hat = best
    test_hat = apply_scalar_direction(pred_t, direction_t, lam)
    return HeadResult(name, val_hat, test_hat, {"ridge": ridge, "shrink": shrink, "lambda_mean": float(np.mean(lam))})


def fit_ridge_stack(features_v: list[np.ndarray], y_v: np.ndarray, ridge: float) -> np.ndarray:
    pred_len = y_v.shape[1]
    k = len(features_v) + 1
    coeff = np.zeros((pred_len, k), dtype=np.float64)
    eye = np.eye(k, dtype=np.float64)
    eye[-1, -1] = 0.0
    for h in range(pred_len):
        cols = [f[:, h, :].reshape(-1).astype(np.float64) for f in features_v]
        y = y_v[:, h, :].reshape(-1).astype(np.float64)
        x = np.stack(cols + [np.ones_like(y)], axis=1)
        coeff[h] = np.linalg.solve(x.T @ x + ridge * eye, x.T @ y)
    return coeff


def apply_ridge_stack(features: list[np.ndarray], coeff: np.ndarray) -> np.ndarray:
    out = np.zeros_like(features[0], dtype=np.float64)
    for h, c in enumerate(coeff):
        y = np.zeros(features[0][:, h, :].size, dtype=np.float64)
        for j, f in enumerate(features):
            y += c[j] * f[:, h, :].reshape(-1).astype(np.float64)
        y += c[-1]
        out[:, h, :] = y.reshape(features[0][:, h, :].shape)
    return out.astype(np.float32)


def fit_best_ridge_stack(
    name: str,
    features_v: list[np.ndarray],
    features_t: list[np.ndarray],
    y_v: np.ndarray,
    ridges: Iterable[float],
) -> HeadResult:
    best = None
    for ridge in ridges:
        coeff = fit_ridge_stack(features_v, y_v, ridge)
        val_hat = apply_ridge_stack(features_v, coeff)
        vm, _ = mse_mae(y_v, val_hat)
        if best is None or vm < best[0]:
            best = (vm, ridge, coeff, val_hat)
    assert best is not None
    _, ridge, coeff, val_hat = best
    test_hat = apply_ridge_stack(features_t, coeff)
    return HeadResult(name, val_hat, test_hat, {"ridge": ridge})


def fit_generic_mlp_readout(
    name: str,
    features_v: list[np.ndarray],
    features_t: list[np.ndarray],
    y_v: np.ndarray,
    hidden: int = 16,
    alpha: float = 1e-3,
    max_samples: int = 20000,
    random_state: int = 0,
) -> HeadResult:
    """Fit a generic nonlinear readout over the same forecast basis as GPSR.

    This is an ordinary stacking control: it receives the same scalar
    coordinates as GPSR but ignores the graph-periodic linear surplus structure.
    For speed and determinism, the baseline uses a fixed random ReLU hidden
    layer and a ridge-fitted output layer, i.e. a one-hidden-layer MLP readout
    with only the last layer optimized on validation labels.
    """
    pred_len = y_v.shape[1]
    out_v = np.zeros_like(y_v, dtype=np.float32)
    out_t = np.zeros_like(features_t[0], dtype=np.float32)
    rng = np.random.default_rng(random_state)
    n_train_total = 0
    ridge = float(alpha)
    for h in range(pred_len):
        x_v = np.stack([f[:, h, :].reshape(-1).astype(np.float64) for f in features_v], axis=1)
        target_v = y_v[:, h, :].reshape(-1).astype(np.float64)
        if x_v.shape[0] > max_samples:
            idx = rng.choice(x_v.shape[0], size=max_samples, replace=False)
            x_fit = x_v[idx]
            y_fit = target_v[idx]
        else:
            x_fit = x_v
            y_fit = target_v
        n_train_total += int(x_fit.shape[0])
        mean = x_fit.mean(axis=0, keepdims=True)
        std = x_fit.std(axis=0, keepdims=True) + 1e-6
        h_rng = np.random.default_rng(random_state + h)
        w_hidden = h_rng.normal(scale=1.0 / np.sqrt(x_fit.shape[1]), size=(x_fit.shape[1], hidden))
        b_hidden = h_rng.normal(scale=0.1, size=(hidden,))

        def design(x: np.ndarray) -> np.ndarray:
            xs = (x - mean) / std
            relu = np.maximum(xs @ w_hidden + b_hidden, 0.0)
            return np.concatenate([xs, relu, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)

        phi = design(x_fit)
        eye = np.eye(phi.shape[1], dtype=np.float64)
        eye[-1, -1] = 0.0
        coef = np.linalg.solve(phi.T @ phi + ridge * eye, phi.T @ y_fit)
        out_v[:, h, :] = (design(x_v) @ coef).reshape(y_v[:, h, :].shape).astype(np.float32)
        x_t = np.stack([f[:, h, :].reshape(-1).astype(np.float64) for f in features_t], axis=1)
        out_t[:, h, :] = (design(x_t) @ coef).reshape(features_t[0][:, h, :].shape).astype(np.float32)
    return HeadResult(
        name,
        out_v,
        out_t,
        {"hidden": hidden, "alpha": alpha, "max_samples": max_samples, "fit_points": n_train_total},
    )


def fit_residual_ridge(
    pred_v: np.ndarray,
    pred_t: np.ndarray,
    dirs_v: list[np.ndarray],
    dirs_t: list[np.ndarray],
    y_v: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_len = pred_v.shape[1]
    k = len(dirs_v)
    coeff = np.zeros((pred_len, k), dtype=np.float64)
    eye = np.eye(k, dtype=np.float64)
    val_out = pred_v.astype(np.float64).copy()
    test_out = pred_t.astype(np.float64).copy()
    for h in range(pred_len):
        cols = [d[:, h, :].reshape(-1).astype(np.float64) for d in dirs_v]
        x = np.stack(cols, axis=1)
        r = (y_v[:, h, :] - pred_v[:, h, :]).reshape(-1).astype(np.float64)
        c = np.linalg.solve(x.T @ x + ridge * eye, x.T @ r)
        coeff[h] = c
        val_delta = sum(c[j] * dirs_v[j][:, h, :] for j in range(k))
        test_delta = sum(c[j] * dirs_t[j][:, h, :] for j in range(k))
        val_out[:, h, :] += val_delta
        test_out[:, h, :] += test_delta
    return val_out.astype(np.float32), test_out.astype(np.float32), coeff


def fit_best_residual_ridge(
    pred_v: np.ndarray,
    pred_t: np.ndarray,
    dirs_v: list[np.ndarray],
    dirs_t: list[np.ndarray],
    y_v: np.ndarray,
    ridges: Iterable[float],
) -> HeadResult:
    best = None
    for ridge in ridges:
        val_hat, test_hat, coeff = fit_residual_ridge(pred_v, pred_t, dirs_v, dirs_t, y_v, ridge)
        vm, _ = mse_mae(y_v, val_hat)
        if best is None or vm < best[0]:
            best = (vm, ridge, val_hat, test_hat, coeff)
    assert best is not None
    _, ridge, val_hat, test_hat, coeff = best
    return HeadResult("BINTS+GenericResidualRidge", val_hat, test_hat, {"ridge": ridge, "coef_abs_mean": float(np.mean(np.abs(coeff)))})


def orthogonalize_pair(
    direction_v: np.ndarray,
    direction_t: np.ndarray,
    basis_v: list[np.ndarray],
    basis_t: list[np.ndarray],
    ridge: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    if not basis_v:
        return direction_v.copy(), direction_t.copy()
    pred_len = direction_v.shape[1]
    out_v = direction_v.astype(np.float64).copy()
    out_t = direction_t.astype(np.float64).copy()
    k = len(basis_v)
    for h in range(pred_len):
        b = [x[:, h, :].reshape(-1).astype(np.float64) for x in basis_v]
        B = np.stack(b, axis=1)
        d = direction_v[:, h, :].reshape(-1).astype(np.float64)
        coef = np.linalg.solve(B.T @ B + ridge * np.eye(k), B.T @ d)
        proj_v = sum(coef[j] * basis_v[j][:, h, :] for j in range(k))
        proj_t = sum(coef[j] * basis_t[j][:, h, :] for j in range(k))
        out_v[:, h, :] -= proj_v
        out_t[:, h, :] -= proj_t
    return out_v.astype(np.float32), out_t.astype(np.float32)


def fit_ancestor_route(
    pred_v: np.ndarray,
    pred_t: np.ndarray,
    y_v: np.ndarray,
    directions_v: dict[str, np.ndarray],
    directions_t: dict[str, np.ndarray],
    order: list[str],
    ridges: Iterable[float],
    shrinks: Iterable[float],
    gate_eps: float,
    norm_eps: float = 1e-10,
) -> HeadResult:
    current_v = pred_v.astype(np.float64).copy()
    current_t = pred_t.astype(np.float64).copy()
    basis_v: list[np.ndarray] = []
    basis_t: list[np.ndarray] = []
    accepted: list[dict[str, object]] = []
    parent_mse, _ = mse_mae(y_v, pred_v)
    current_mse = parent_mse
    for name in order:
        dv_perp, dt_perp = orthogonalize_pair(directions_v[name], directions_t[name], basis_v, basis_t)
        if float(np.mean(dv_perp.astype(np.float64) ** 2)) <= norm_eps:
            continue
        best = None
        for ridge in ridges:
            for shrink in shrinks:
                lam = fit_scalar_direction(pred_v, dv_perp, y_v, ridge, shrink)
                val_hat = current_v + np.asarray(apply_scalar_direction(np.zeros_like(pred_v), dv_perp, lam), dtype=np.float64)
                vm, _ = mse_mae(y_v, val_hat)
                if best is None or vm < best[0]:
                    best = (vm, ridge, shrink, lam, val_hat)
        assert best is not None
        vm, ridge, shrink, lam, val_hat = best
        surplus = current_mse - vm
        if surplus > gate_eps:
            delta_t = apply_scalar_direction(np.zeros_like(pred_t), dt_perp, lam).astype(np.float64)
            current_v = val_hat.astype(np.float64)
            current_t = current_t + delta_t
            current_mse = vm
            basis_v.append(dv_perp)
            basis_t.append(dt_perp)
            accepted.append(
                {
                    "edge": name,
                    "ridge": ridge,
                    "shrink": shrink,
                    "val_surplus": float(surplus),
                    "lambda_mean": float(np.mean(lam)),
                }
            )
    return HeadResult("BINTS+MCPF-AO-Route", current_v.astype(np.float32), current_t.astype(np.float32), {"accepted": accepted, "gate_eps": gate_eps})


def split_shift_certificate(y_v: np.ndarray, parent_v: np.ndarray, heads: list[HeadResult]) -> float:
    n = y_v.shape[0]
    if n < 4:
        return 0.0
    cut = n // 2
    vals = []
    for head in heads:
        p1, _ = mse_mae(y_v[:cut], parent_v[:cut])
        h1, _ = mse_mae(y_v[:cut], head.val_pred[:cut])
        p2, _ = mse_mae(y_v[cut:], parent_v[cut:])
        h2, _ = mse_mae(y_v[cut:], head.val_pred[cut:])
        vals.append(abs((p1 - h1) - (p2 - h2)))
    return float(max(vals) if vals else 0.0)


def select_by_validation_metric(
    name: str,
    parent_v: np.ndarray,
    parent_t: np.ndarray,
    y_v: np.ndarray,
    heads: list[HeadResult],
    metric: str,
) -> HeadResult:
    """Select a deployable head by validation MSE or MAE, including parent fallback."""
    parent = HeadResult("BINTS", parent_v, parent_t, {})
    candidates = [parent] + heads
    if metric == "mse":
        scorer = lambda h: mse_mae(y_v, h.val_pred)[0]
    elif metric == "mae":
        scorer = lambda h: mse_mae(y_v, h.val_pred)[1]
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    best = min(candidates, key=scorer)
    return HeadResult(
        name,
        best.val_pred,
        best.test_pred,
        {"selected": best.method, "metric": metric, "val_score": float(scorer(best))},
    )


def split_metric_instability(
    y_v: np.ndarray,
    parent_v: np.ndarray,
    head_v: np.ndarray,
    metric: str,
) -> float:
    """Difference between first-half and second-half validation improvements."""
    n = y_v.shape[0]
    if n < 4:
        return 0.0
    cut = n // 2
    if metric == "mse":
        metric_fn = lambda y, p: mse_mae(y, p)[0]
    elif metric == "mae":
        metric_fn = lambda y, p: mse_mae(y, p)[1]
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    imp_1 = metric_fn(y_v[:cut], parent_v[:cut]) - metric_fn(y_v[:cut], head_v[:cut])
    imp_2 = metric_fn(y_v[cut:], parent_v[cut:]) - metric_fn(y_v[cut:], head_v[cut:])
    return float(abs(imp_1 - imp_2))


def select_by_penalized_validation_metric(
    name: str,
    parent_v: np.ndarray,
    parent_t: np.ndarray,
    y_v: np.ndarray,
    heads: list[HeadResult],
    metric: str,
    penalty: float,
) -> HeadResult:
    """Validation selector with a split-stability penalty and parent fallback."""
    parent = HeadResult("BINTS", parent_v, parent_t, {})
    candidates = [parent] + heads
    if metric == "mse":
        metric_fn = lambda h: mse_mae(y_v, h.val_pred)[0]
    elif metric == "mae":
        metric_fn = lambda h: mse_mae(y_v, h.val_pred)[1]
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    def score(head: HeadResult) -> float:
        return float(metric_fn(head) + penalty * split_metric_instability(y_v, parent_v, head.val_pred, metric))

    best = min(candidates, key=score)
    return HeadResult(
        name,
        best.val_pred,
        best.test_pred,
        {
            "selected": best.method,
            "metric": metric,
            "penalty": penalty,
            "penalized_score": score(best),
            "val_score": float(metric_fn(best)),
            "split_instability": split_metric_instability(y_v, parent_v, best.val_pred, metric),
        },
    )


def select_metric_aware_gpsr_prc(
    name: str,
    parent_v: np.ndarray,
    parent_t: np.ndarray,
    y_v: np.ndarray,
    heads: list[HeadResult],
    metric: str,
) -> HeadResult:
    """Metric-aware deployment rule: GPSR family for MSE, PRC/fallback for MAE."""
    by_name = {h.method: h for h in heads}
    parent = HeadResult("BINTS", parent_v, parent_t, {})
    if metric == "mse":
        candidates = [parent]
        for method in ["BINTS+MCPF-GPSR-MLP", "BINTS+MCPF-GPSR"]:
            if method in by_name:
                candidates.append(by_name[method])
        metric_fn = lambda h: mse_mae(y_v, h.val_pred)[0]
    elif metric == "mae":
        candidates = [parent]
        if "BINTS+MCPF-PRC" in by_name:
            candidates.append(by_name["BINTS+MCPF-PRC"])
        metric_fn = lambda h: mse_mae(y_v, h.val_pred)[1]
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    best = min(candidates, key=metric_fn)
    return HeadResult(
        name,
        best.val_pred,
        best.test_pred,
        {"selected": best.method, "metric": metric, "candidate_policy": "mse:gpsr/gpsr-mlp;mae:prc/fallback", "val_score": float(metric_fn(best))},
    )


def write_rows(rows: list[dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relabel_method(name: str, parent_name: str) -> str:
    if parent_name == "BINTS":
        return name
    if name == "BINTS":
        return parent_name
    if name.startswith("BINTS+"):
        return parent_name + name[len("BINTS"):]
    return name


def relabel_detail(detail: dict[str, object], parent_name: str) -> dict[str, object]:
    if parent_name == "BINTS":
        return detail
    out: dict[str, object] = {}
    for key, value in detail.items():
        if isinstance(value, str):
            out[key] = relabel_method(value, parent_name)
        else:
            out[key] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--parent-name", default="BINTS")
    ap.add_argument("--cycle", type=int, default=24)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--nodes", type=int, default=None)
    ap.add_argument("--adj-root", default="external/baseline_sources/BINTS/datasets")
    ap.add_argument("--adj-path", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-eps", type=float, default=1e-8)
    ap.add_argument("--norm-eps", type=float, default=1e-10)
    ap.add_argument("--graph-mode", choices=["real", "identity", "permute", "random"], default="real")
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--skip-mlp", action="store_true", help="Skip fixed-feature nonlinear readouts for large views.")
    args = ap.parse_args()

    z = np.load(args.npz)
    val_pred = z["val_pred"].astype(np.float32)
    val_y = z["val_y"].astype(np.float32)
    val_x = z["val_x"].astype(np.float32)
    test_pred = z["test_pred"].astype(np.float32)
    test_y = z["test_y"].astype(np.float32)
    test_x = z["test_x"].astype(np.float32)
    pred_len = test_y.shape[1]

    val_anchor = periodic_anchor(val_x, pred_len, args.cycle).astype(np.float32)
    test_anchor = periodic_anchor(test_x, pred_len, args.cycle).astype(np.float32)
    adj_root = Path(args.adj_root) if args.adj_root else None
    w, nodes, adj_source = load_adjacency(
        args.dataset,
        adj_root,
        test_y.shape[-1],
        args.nodes,
        args.graph_mode,
        args.graph_seed,
        Path(args.adj_path) if args.adj_path else None,
    )

    d_t_v = val_anchor - val_pred
    d_t_t = test_anchor - test_pred
    d_g_v = graph_filter(d_t_v, w, nodes)
    d_g_t = graph_filter(d_t_t, w, nodes)
    d_c_v = d_t_v - d_g_v
    d_c_t = d_t_t - d_g_t
    graph_anchor_v = val_pred + d_g_v
    graph_anchor_t = test_pred + d_g_t
    contrast_anchor_v = val_pred + d_c_v
    contrast_anchor_t = test_pred + d_c_t

    ridges = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    scalar_ridges = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    shrinks = [0.25, 0.5, 0.75, 1.0]

    heads: list[HeadResult] = []
    aff = fit_affine_horizon(val_pred, val_y)
    heads.append(HeadResult("BINTS+AffineH", apply_affine_horizon(val_pred, aff), apply_affine_horizon(test_pred, aff), {}))
    heads.append(fit_best_scalar_head("BINTS+MCPF-PRC", val_pred, test_pred, d_t_v, d_t_t, val_y, scalar_ridges, shrinks))
    heads.append(fit_best_scalar_head("BINTS+MCPF-GAR", val_pred, test_pred, d_g_v, d_g_t, val_y, scalar_ridges, shrinks))
    heads.append(fit_best_scalar_head("BINTS+MCPF-GCR", val_pred, test_pred, d_c_v, d_c_t, val_y, scalar_ridges, shrinks))
    heads.append(fit_best_ridge_stack("BINTS+MCPF-PAR", [val_pred, val_anchor], [test_pred, test_anchor], val_y, ridges))
    heads.append(fit_best_residual_ridge(val_pred, test_pred, [d_t_v, d_g_v, d_c_v], [d_t_t, d_g_t, d_c_t], val_y, ridges))
    heads.append(
        fit_best_ridge_stack(
            "BINTS+MCPF-GPSR",
            [val_pred, val_anchor, graph_anchor_v, contrast_anchor_v],
            [test_pred, test_anchor, graph_anchor_t, contrast_anchor_t],
            val_y,
            ridges,
        )
    )
    heads.append(
        fit_ancestor_route(
            val_pred,
            test_pred,
            val_y,
            {"PRC": d_t_v, "GAR": d_g_v, "GCR": d_c_v},
            {"PRC": d_t_t, "GAR": d_g_t, "GCR": d_c_t},
            ["PRC", "GAR", "GCR"],
            scalar_ridges,
            shrinks,
            args.gate_eps,
            args.norm_eps,
        )
    )
    if not args.skip_mlp:
        heads.append(
            fit_generic_mlp_readout(
                "BINTS+GenericMLPStack",
                [val_pred, val_anchor],
                [test_pred, test_anchor],
                val_y,
            )
        )
        heads.append(
            fit_generic_mlp_readout(
                "BINTS+MCPF-GPSR-MLP",
                [val_pred, val_anchor, graph_anchor_v, contrast_anchor_v],
                [test_pred, test_anchor, graph_anchor_t, contrast_anchor_t],
                val_y,
            )
        )
    base_heads = list(heads)
    heads.append(select_by_validation_metric("BINTS+MCPF-Select-MSE", val_pred, test_pred, val_y, base_heads, "mse"))
    heads.append(select_by_validation_metric("BINTS+MCPF-Select-MAE", val_pred, test_pred, val_y, base_heads, "mae"))
    heads.append(select_by_penalized_validation_metric("BINTS+MCPF-Stable-Select-MAE", val_pred, test_pred, val_y, base_heads, "mae", penalty=1.0))
    heads.append(select_metric_aware_gpsr_prc("BINTS+MCPF-MetricAware-MSE", val_pred, test_pred, val_y, base_heads, "mse"))
    heads.append(select_metric_aware_gpsr_prc("BINTS+MCPF-MetricAware-MAE", val_pred, test_pred, val_y, base_heads, "mae"))

    parent_val_mse, parent_val_mae = mse_mae(val_y, val_pred)
    parent_mse, parent_mae = mse_mae(test_y, test_pred)
    eps_split = split_shift_certificate(val_y, val_pred, heads)
    rows: list[dict[str, object]] = [
        {
            "dataset": args.dataset,
            "npz": str(args.npz),
            "method": relabel_method("BINTS", args.parent_name),
            "val_mse": parent_val_mse,
            "val_mae": parent_val_mae,
            "mse": parent_mse,
            "mae": parent_mae,
            "delta_mse_vs_bints": 0.0,
            "delta_mae_vs_bints": 0.0,
            "delta_mse_vs_parent": 0.0,
            "delta_mae_vs_parent": 0.0,
            "val_surplus_mse": 0.0,
            "split_surplus_shift": eps_split,
            "nodes": nodes,
            "graph_mode": args.graph_mode,
            "adjacency": adj_source,
            "detail": "{}",
        }
    ]
    for head in heads:
        val_m, val_a = mse_mae(val_y, head.val_pred)
        m, a = mse_mae(test_y, head.test_pred)
        rows.append(
            {
                "dataset": args.dataset,
                "npz": str(args.npz),
                "method": relabel_method(head.method, args.parent_name),
                "val_mse": val_m,
                "val_mae": val_a,
                "mse": m,
                "mae": a,
                "delta_mse_vs_bints": m - parent_mse,
                "delta_mae_vs_bints": a - parent_mae,
                "delta_mse_vs_parent": m - parent_mse,
                "delta_mae_vs_parent": a - parent_mae,
                "val_surplus_mse": parent_val_mse - val_m,
                "split_surplus_shift": eps_split,
                "nodes": nodes,
                "graph_mode": args.graph_mode,
                "adjacency": adj_source,
                "detail": json.dumps(relabel_detail(head.detail, args.parent_name), sort_keys=True),
            }
        )

    out = Path(args.out)
    write_rows(rows, out)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
