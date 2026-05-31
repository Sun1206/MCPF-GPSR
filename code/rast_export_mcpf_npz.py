"""Export RAST validation/test forecasts to the MCPF calibration NPZ format.

The exported file contains six arrays:
  val_pred, val_y, val_x, test_pred, test_y, test_x

RAST/BasicTS stores tensors as [sample, horizon, node, channel].  The MCPF
calibrator consumes [sample, horizon, flattened_dimension], so this exporter
flattens the node/channel axes after inverse scaling.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def flatten_last_axes(x: np.ndarray) -> np.ndarray:
    if x.ndim == 4:
        return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected a 3D or 4D array, got shape {x.shape}")


@torch.no_grad()
def collect_split(runner, loader, desc: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    inputs: list[np.ndarray] = []
    runner.model.eval()
    for iter_index, data in enumerate(tqdm(loader, desc=desc)):
        out = runner.forward(data, epoch=None, iter_num=iter_index, train=False)
        preds.append(out["prediction"].detach().cpu().numpy())
        targets.append(out["target"].detach().cpu().numpy())
        inputs.append(out["inputs"].detach().cpu().numpy())
    pred = flatten_last_axes(np.concatenate(preds, axis=0)).astype(np.float32)
    target = flatten_last_axes(np.concatenate(targets, axis=0)).astype(np.float32)
    hist = flatten_last_axes(np.concatenate(inputs, axis=0)).astype(np.float32)
    return pred, target, hist


def infer_best_checkpoint(runner) -> str:
    metric = runner.target_metrics.replace("/", "_")
    path = Path(runner.ckpt_save_dir) / f"{runner.model_name}_best_val_{metric}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {path}")
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rast-root", default="external/baseline_sources/RAST")
    ap.add_argument("--config", required=True, help="RAST config, e.g. src/scripts/main/RAST_PEMS04.py")
    ap.add_argument("--checkpoint", default="", help="Optional checkpoint path. Defaults to the best validation checkpoint.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--device-type", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    rast_root = Path(args.rast_root).resolve()
    sys.path.insert(0, str(rast_root))
    os.chdir(rast_root)

    sys.argv = [sys.argv[0]]
    from easytorch.config import init_cfg  # pylint: disable=import-error,import-outside-toplevel
    from easytorch.device import set_device_type  # pylint: disable=import-error,import-outside-toplevel
    from easytorch.utils import get_logger, set_visible_devices  # pylint: disable=import-error,import-outside-toplevel

    set_device_type(args.device_type)
    if args.device_type != "cpu":
        set_visible_devices(args.gpus)

    cfg = init_cfg(args.config, save=True)
    if args.batch_size is not None:
        cfg.VAL.DATA.BATCH_SIZE = args.batch_size
        cfg.TEST.DATA.BATCH_SIZE = args.batch_size

    logger = get_logger("rast-mcpf-export")
    runner = cfg["RUNNER"](cfg)
    runner.init_logger(logger_name="rast-mcpf-export", log_file_name="mcpf_export_log")
    runner.init_validation(cfg)
    runner.init_test(cfg)

    ckpt = args.checkpoint or infer_best_checkpoint(runner)
    logger.info("Loading checkpoint %s", ckpt)
    runner.load_model(ckpt_path=ckpt, strict=True)

    val_pred, val_y, val_x = collect_split(runner, runner.val_data_loader, "valid")
    test_pred, test_y, test_x = collect_split(runner, runner.test_data_loader, "test")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        val_pred=val_pred,
        val_y=val_y,
        val_x=val_x,
        test_pred=test_pred,
        test_y=test_y,
        test_x=test_x,
    )
    print(f"Saved {out}")
    print(f"val_pred={val_pred.shape} val_y={val_y.shape} val_x={val_x.shape}")
    print(f"test_pred={test_pred.shape} test_y={test_y.shape} test_x={test_x.shape}")


if __name__ == "__main__":
    main()
