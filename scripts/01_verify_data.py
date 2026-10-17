"""Verify raw files against data/checksums.json (sha256)."""

import json
import sys
from pathlib import Path

from tkgat.registry import sha256_file

expected = json.loads(Path("data/checksums.json").read_text())
bad = 0
for rel, digest in expected.items():
    p = Path("data/raw") / rel
    if not p.exists():
        print(f"MISSING  {rel}")
        bad += 1
    elif sha256_file(p) != digest:
        print(f"MISMATCH {rel}")
        bad += 1
    else:
        print(f"ok       {rel}")
sys.exit(1 if bad else 0)
