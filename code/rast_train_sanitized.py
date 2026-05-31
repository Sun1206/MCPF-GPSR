"""Launch RAST training without leaking wrapper CLI args into EasyTorch."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rast-root", default="external/baseline_sources/RAST")
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpus", default="0")
    args = ap.parse_args()

    rast_root = Path(args.rast_root).resolve()
    sys.path.insert(0, str(rast_root))
    os.chdir(rast_root)

    sys.argv = [sys.argv[0]]
    import basicts  # pylint: disable=import-error,import-outside-toplevel

    basicts.launch_training(args.config, args.gpus, node_rank=0)


if __name__ == "__main__":
    main()
