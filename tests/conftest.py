from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic() -> pd.DataFrame:
    """60 users x 40 items over 100 days, deterministic."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(60):
        for i in rng.choice(40, size=25, replace=False):
            rows.append((u, int(i), int(rng.integers(1, 6)), int(rng.integers(0, 100 * 86400))))
    return pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])


def data_available(name: str) -> bool:
    return (ROOT / "data" / "raw" / name).exists()
