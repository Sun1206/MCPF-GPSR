"""Aggregate extended BINTS residual-calibration diagnostics.

The extended calibrator evaluates graph-anchor residuals (GAR),
graph-contrast residuals (GCR), ancestor-orthogonal routing, and generic
residual/stacking controls from frozen BINTS predictions.  This analyzer writes
paper-ready CSVs with paired win counts and relative deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "bints_runs" / "extended_ao_v4"
OUT = Path(__file__).resolve().parent / "bints_runs" / "analysis_extended_ao"


def parse_meta(path: Path) -> tuple[str, int, int]:
    """Return dataset, seed, and prediction horizon from an extended CSV name."""
    m = re.search(r"__([a-z_]+)_seq4_pred(\d+)_khop5_extended\.csv$", path.name)
    if not m:
        m = re.search(r"([a-z_]+)_seq4_pred(\d+)_khop5_extended\.csv$", path.name)
    if not m:
        raise ValueError(f"Cannot parse dataset/horizon from {path.name}")
    dataset = m.group(1)
    horizon = int(m.group(2))

    sm = re.search(r"seed(\d+)", path.name)
    if sm:
        seed = int(sm.group(1))
    elif "__server_" in path.name:
        seed = 0
    else:
        seed = 0
    return dataset, seed, horizon


def mean_std(xs: list[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def read_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(ROOT.glob("*_extended.csv")):
        dataset, seed, horizon = parse_meta(path)
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                detail = {}
                if r.get("detail"):
                    try:
                        detail = json.loads(r["detail"])
                    except json.JSONDecodeError:
                        detail = {}
                accepted = detail.get("accepted", [])
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "horizon": horizon,
                        "method": r["method"],
                        "mse": float(r["mse"]),
                        "mae": float(r["mae"]),
                        "val_mse": float(r["val_mse"]),
                        "val_mae": float(r["val_mae"]),
                        "val_surplus_mse": float(r["val_surplus_mse"]),
                        "split_surplus_shift": float(r["split_surplus_shift"]),
                        "accepted_edges": "+".join(a.get("edge", "") for a in accepted) if accepted else "",
                        "accepted_count": len(accepted),
                        "source_file": str(path),
                    }
                )
    if not rows:
        raise SystemExit(f"No extended CSV files found under {ROOT}")

    parent = {
        (r["dataset"], r["seed"], r["horizon"]): r
        for r in rows
        if r["method"] == "BINTS"
    }
    gpsr = {
        (r["dataset"], r["seed"], r["horizon"]): r
        for r in rows
        if r["method"] == "BINTS+MCPF-GPSR"
    }
    for r in rows:
        key = (r["dataset"], r["seed"], r["horizon"])
        b = parent[key]
        g = gpsr.get(key)
        r["delta_mse_vs_bints"] = float(r["mse"]) - float(b["mse"])
        r["delta_mae_vs_bints"] = float(r["mae"]) - float(b["mae"])
        r["rel_mse_pct_vs_bints"] = 100.0 * float(r["delta_mse_vs_bints"]) / float(b["mse"])
        r["rel_mae_pct_vs_bints"] = 100.0 * float(r["delta_mae_vs_bints"]) / float(b["mae"])
        r["mse_win_vs_bints"] = int(float(r["mse"]) < float(b["mse"]))
        r["mae_win_vs_bints"] = int(float(r["mae"]) < float(b["mae"]))
        if g is not None:
            r["delta_mse_vs_gpsr"] = float(r["mse"]) - float(g["mse"])
            r["delta_mae_vs_gpsr"] = float(r["mae"]) - float(g["mae"])
            r["mse_win_vs_gpsr"] = int(float(r["mse"]) < float(g["mse"]))
            r["mae_win_vs_gpsr"] = int(float(r["mae"]) < float(g["mae"]))
        else:
            r["delta_mse_vs_gpsr"] = ""
            r["delta_mae_vs_gpsr"] = ""
            r["mse_win_vs_gpsr"] = ""
            r["mae_win_vs_gpsr"] = ""
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        grouped[tuple(r[k] for k in keys)].append(r)
    out: list[dict[str, object]] = []
    for key, rs in sorted(grouped.items()):
        rec = {k: v for k, v in zip(keys, key)}
        for metric in ["mse", "mae", "rel_mse_pct_vs_bints", "rel_mae_pct_vs_bints", "val_surplus_mse", "split_surplus_shift"]:
            m, s = mean_std([float(r[metric]) for r in rs])
            rec[f"{metric}_mean"] = m
            rec[f"{metric}_std"] = s
        rec["n"] = len(rs)
        rec["mse_wins_vs_bints"] = sum(int(r["mse_win_vs_bints"]) for r in rs)
        rec["mae_wins_vs_bints"] = sum(int(r["mae_win_vs_bints"]) for r in rs)
        if any(r["mse_win_vs_gpsr"] != "" for r in rs):
            rec["mse_wins_vs_gpsr"] = sum(int(r["mse_win_vs_gpsr"]) for r in rs if r["mse_win_vs_gpsr"] != "")
            rec["mae_wins_vs_gpsr"] = sum(int(r["mae_win_vs_gpsr"]) for r in rs if r["mae_win_vs_gpsr"] != "")
        accepted = [str(r["accepted_edges"]) for r in rs if r["accepted_edges"]]
        rec["accepted_edge_patterns"] = ";".join(sorted(set(accepted)))
        out.append(rec)
    return out


def best_by_cell(rows: list[dict[str, object]], candidate_methods: set[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        if r["method"] in candidate_methods:
            grouped[(str(r["dataset"]), int(r["seed"]), int(r["horizon"]))].append(r)
    out: list[dict[str, object]] = []
    for (dataset, seed, horizon), rs in sorted(grouped.items()):
        best_mse = min(rs, key=lambda r: float(r["mse"]))
        best_mae = min(rs, key=lambda r: float(r["mae"]))
        parent = next(r for r in rows if r["dataset"] == dataset and r["seed"] == seed and r["horizon"] == horizon and r["method"] == "BINTS")
        out.append(
            {
                "dataset": dataset,
                "seed": seed,
                "horizon": horizon,
                "best_mse_method": best_mse["method"],
                "best_mse": best_mse["mse"],
                "best_mse_delta_vs_bints": float(best_mse["mse"]) - float(parent["mse"]),
                "best_mae_method": best_mae["method"],
                "best_mae": best_mae["mae"],
                "best_mae_delta_vs_bints": float(best_mae["mae"]) - float(parent["mae"]),
            }
        )
    return out


def sign_p_value(wins: int, n: int) -> float:
    """One-sided exact sign-test p-value for improvement over a paired baseline."""
    if n <= 0:
        return float("nan")
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n))


def bootstrap_ci(xs: list[float], iters: int = 20000, seed: int = 7) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(iters, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pairwise_tests(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Paired sign tests and bootstrap CIs over seed-horizon cells."""
    by_cell_method = {
        (str(r["dataset"]), int(r["seed"]), int(r["horizon"]), str(r["method"])): r
        for r in rows
    }
    datasets = sorted({str(r["dataset"]) for r in rows})
    comparisons = [
        ("BINTS+MCPF-GPSR", "BINTS"),
        ("BINTS+MCPF-GPSR", "BINTS+MCPF-PAR"),
        ("BINTS+MCPF-GPSR", "BINTS+GenericResidualRidge"),
        ("BINTS+MCPF-GPSR", "BINTS+GenericMLPStack"),
        ("BINTS+MCPF-GPSR-MLP", "BINTS+GenericMLPStack"),
        ("BINTS+MCPF-GPSR-MLP", "BINTS"),
        ("BINTS+MCPF-PAR", "BINTS"),
        ("BINTS+GenericMLPStack", "BINTS"),
    ]
    out: list[dict[str, object]] = []
    for dataset in datasets + ["all"]:
        cells = sorted(
            {
                (str(r["dataset"]), int(r["seed"]), int(r["horizon"]))
                for r in rows
                if dataset == "all" or str(r["dataset"]) == dataset
            }
        )
        for method, baseline in comparisons:
            for metric in ["mse", "mae"]:
                deltas: list[float] = []
                rels: list[float] = []
                for d, seed, horizon in cells:
                    a = by_cell_method.get((d, seed, horizon, method))
                    b = by_cell_method.get((d, seed, horizon, baseline))
                    if a is None or b is None:
                        continue
                    delta = float(a[metric]) - float(b[metric])
                    deltas.append(delta)
                    rels.append(100.0 * delta / float(b[metric]))
                if not deltas:
                    continue
                wins = sum(1 for x in deltas if x < 0.0)
                ci_lo, ci_hi = bootstrap_ci(deltas)
                rel_lo, rel_hi = bootstrap_ci(rels)
                out.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "baseline": baseline,
                        "metric": metric,
                        "n": len(deltas),
                        "wins": wins,
                        "mean_delta": float(np.mean(deltas)),
                        "ci95_delta_low": ci_lo,
                        "ci95_delta_high": ci_hi,
                        "mean_rel_pct": float(np.mean(rels)),
                        "ci95_rel_pct_low": rel_lo,
                        "ci95_rel_pct_high": rel_hi,
                        "sign_test_p_one_sided": sign_p_value(wins, len(deltas)),
                    }
                )
    return out


def main() -> None:
    global ROOT, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT), help="Directory containing *_extended.csv files.")
    ap.add_argument("--out", default=str(OUT), help="Output directory for summary CSVs.")
    args = ap.parse_args()
    ROOT = Path(args.root).resolve()
    OUT = Path(args.out).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    write_csv(OUT / "extended_rows.csv", rows)
    write_csv(OUT / "extended_by_dataset_horizon_method.csv", summarize(rows, ("dataset", "horizon", "method")))
    write_csv(OUT / "extended_by_dataset_method.csv", summarize(rows, ("dataset", "method")))
    write_csv(OUT / "extended_by_method.csv", summarize(rows, ("method",)))
    write_csv(
        OUT / "extended_best_structured_vs_generic.csv",
        best_by_cell(
            rows,
            {
                "BINTS+MCPF-PRC",
                "BINTS+MCPF-GAR",
                "BINTS+MCPF-GCR",
                "BINTS+MCPF-PAR",
                "BINTS+MCPF-AO-Route",
                "BINTS+GenericResidualRidge",
                "BINTS+GenericMLPStack",
                "BINTS+MCPF-GPSR-MLP",
                "BINTS+MCPF-GPSR",
                "BINTS+MCPF-Select-MSE",
                "BINTS+MCPF-Select-MAE",
            },
        ),
    )
    write_csv(OUT / "extended_pairwise_tests.csv", pairwise_tests(rows))
    print(f"Wrote summaries to {OUT}")


if __name__ == "__main__":
    main()
