"""Summarize graph-mode ablations for MCPF-GPSR.

The real graph run is the default extended_ao_v4 result.  Identity, permuted,
and random graph runs are stored in graph_ablation_* folders.
"""

from __future__ import annotations

import csv
import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parent / "bints_runs"
DEFAULT_OUT = DEFAULT_ROOT / "analysis_graph_ablation"
ROOT = DEFAULT_ROOT
OUT = DEFAULT_OUT
RUNS = {
    "real": ROOT / "extended_ao_v4",
    "identity": ROOT / "graph_ablation_identity",
    "permute": ROOT / "graph_ablation_permute",
    "random": ROOT / "graph_ablation_random",
}


def configure(root: Path, out: Path | None = None) -> None:
    global ROOT, OUT, RUNS
    ROOT = root
    OUT = out if out is not None else root / "analysis_graph_ablation"
    RUNS = {
        "real": ROOT / "extended_ao_v4",
        "identity": ROOT / "graph_ablation_identity",
        "permute": ROOT / "graph_ablation_permute",
        "random": ROOT / "graph_ablation_random",
    }


def parse_meta(path: Path) -> tuple[str, int, int]:
    m = re.search(r"(covid|nyc)_seq4_pred(\d+)_khop5", path.name)
    if not m:
        raise ValueError(path.name)
    dataset = m.group(1)
    horizon = int(m.group(2))
    sm = re.search(r"__(covid|nyc)_seed(\d+)__", path.name)
    seed = int(sm.group(2)) if sm else 0
    return dataset, seed, horizon


def mean_std(xs: list[float]) -> tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    if a.size <= 1:
        return float(a.mean()), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def read_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode, folder in RUNS.items():
        for path in sorted(folder.glob("*.csv")):
            if path.name.startswith("run_manifest"):
                continue
            dataset, seed, horizon = parse_meta(path)
            with path.open(newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["method"] not in {
                        "BINTS",
                        "BINTS+MCPF-PAR",
                        "BINTS+MCPF-GPSR",
                        "BINTS+MCPF-GAR",
                        "BINTS+MCPF-GCR",
                        "BINTS+MCPF-Select-MSE",
                        "BINTS+MCPF-Select-MAE",
                    }:
                        continue
                    rows.append(
                        {
                            "graph_mode": mode,
                            "dataset": dataset,
                            "seed": seed,
                            "horizon": horizon,
                            "method": r["method"],
                            "mse": float(r["mse"]),
                            "mae": float(r["mae"]),
                            "val_mse": float(r["val_mse"]),
                            "val_mae": float(r["val_mae"]),
                        }
                    )
    parent = {
        (r["graph_mode"], r["dataset"], r["seed"], r["horizon"]): r
        for r in rows
        if r["method"] == "BINTS"
    }
    par = {
        (r["graph_mode"], r["dataset"], r["seed"], r["horizon"]): r
        for r in rows
        if r["method"] == "BINTS+MCPF-PAR"
    }
    for r in rows:
        key = (r["graph_mode"], r["dataset"], r["seed"], r["horizon"])
        b = parent[key]
        p = par[key]
        r["rel_mse_vs_bints"] = 100.0 * (float(r["mse"]) - float(b["mse"])) / float(b["mse"])
        r["rel_mae_vs_bints"] = 100.0 * (float(r["mae"]) - float(b["mae"])) / float(b["mae"])
        r["rel_mse_vs_par"] = 100.0 * (float(r["mse"]) - float(p["mse"])) / float(p["mse"])
        r["rel_mae_vs_par"] = 100.0 * (float(r["mae"]) - float(p["mae"])) / float(p["mae"])
        r["mse_win_vs_bints"] = int(float(r["mse"]) < float(b["mse"]))
        r["mae_win_vs_bints"] = int(float(r["mae"]) < float(b["mae"]))
        r["mse_win_vs_par"] = int(float(r["mse"]) < float(p["mse"]))
        r["mae_win_vs_par"] = int(float(r["mae"]) < float(p["mae"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        grouped[tuple(r[k] for k in keys)].append(r)
    out: list[dict[str, object]] = []
    for key, rs in sorted(grouped.items()):
        rec = {k: v for k, v in zip(keys, key)}
        for metric in ["rel_mse_vs_bints", "rel_mae_vs_bints", "rel_mse_vs_par", "rel_mae_vs_par"]:
            m, s = mean_std([float(r[metric]) for r in rs])
            rec[f"{metric}_mean"] = m
            rec[f"{metric}_std"] = s
        rec["n"] = len(rs)
        rec["mse_wins_vs_bints"] = sum(int(r["mse_win_vs_bints"]) for r in rs)
        rec["mae_wins_vs_bints"] = sum(int(r["mae_win_vs_bints"]) for r in rs)
        rec["mse_wins_vs_par"] = sum(int(r["mse_win_vs_par"]) for r in rs)
        rec["mae_wins_vs_par"] = sum(int(r["mae_win_vs_par"]) for r in rs)
        out.append(rec)
    return out


def sign_p_value(wins: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n))


def bootstrap_ci(xs: list[float], iters: int = 20000, seed: int = 13) -> tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(iters, a.size))
    means = a[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def graph_pairwise_tests(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Paired tests comparing true graph GPSR with wrong/no-graph controls."""
    gpsr = [r for r in rows if r["method"] == "BINTS+MCPF-GPSR"]
    by_cell = {
        (str(r["graph_mode"]), str(r["dataset"]), int(r["seed"]), int(r["horizon"])): r
        for r in gpsr
    }
    out: list[dict[str, object]] = []
    datasets = sorted({str(r["dataset"]) for r in gpsr})
    baselines = ["identity", "permute", "random"]
    for dataset in datasets + ["all"]:
        cells = sorted(
            {
                (str(r["dataset"]), int(r["seed"]), int(r["horizon"]))
                for r in gpsr
                if r["graph_mode"] == "real" and (dataset == "all" or str(r["dataset"]) == dataset)
            }
        )
        for base in baselines:
            for metric in ["mse", "mae"]:
                deltas: list[float] = []
                rels: list[float] = []
                for d, seed, horizon in cells:
                    real = by_cell.get(("real", d, seed, horizon))
                    ctrl = by_cell.get((base, d, seed, horizon))
                    if real is None or ctrl is None:
                        continue
                    delta = float(real[metric]) - float(ctrl[metric])
                    deltas.append(delta)
                    rels.append(100.0 * delta / float(ctrl[metric]))
                if not deltas:
                    continue
                wins = sum(1 for x in deltas if x < 0.0)
                lo, hi = bootstrap_ci(rels)
                out.append(
                    {
                        "dataset": dataset,
                        "method": "BINTS+MCPF-GPSR",
                        "graph": "real",
                        "baseline_graph": base,
                        "metric": metric,
                        "n": len(deltas),
                        "wins": wins,
                        "mean_rel_pct": float(np.mean(rels)),
                        "ci95_rel_pct_low": lo,
                        "ci95_rel_pct_high": hi,
                        "sign_test_p_one_sided": sign_p_value(wins, len(deltas)),
                    }
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    configure(args.root, args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    write_csv(OUT / "graph_ablation_rows.csv", rows)
    write_csv(OUT / "graph_ablation_by_mode_dataset_method.csv", summarize(rows, ("graph_mode", "dataset", "method")))
    write_csv(OUT / "graph_ablation_gpsr_by_mode_dataset.csv", summarize([r for r in rows if r["method"] == "BINTS+MCPF-GPSR"], ("graph_mode", "dataset", "method")))
    write_csv(OUT / "graph_ablation_pairwise_tests.csv", graph_pairwise_tests(rows))
    print(f"Wrote graph ablation summaries to {OUT}")


if __name__ == "__main__":
    main()
