"""Build the temporal split for a data config and refuse to save it if any leakage check fails.

Usage: python scripts/02_build_splits.py --data configs/data/ml100k.yaml
"""

import argparse
import json

from tkgat.pipeline.build import build_split, load_config, save_split

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
args = ap.parse_args()

cfg = load_config(args.data)
split, clean_report = build_split(cfg)
out = save_split(split, cfg, clean_report)
print(json.dumps(split.summary(), indent=2, default=str))
print(f"saved to {out} (all leakage checks passed)")
