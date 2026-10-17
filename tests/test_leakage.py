import pandas as pd
import pytest

from tkgat.data.leakage import check_split
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.pipeline.build import build_split, load_config

from .conftest import ROOT, data_available


def test_synthetic_split_is_clean(synthetic):
    assert check_split(temporal_split(synthetic, SplitConfig(kcore=2))) == []


def test_detects_future_row_in_train(synthetic):
    s = temporal_split(synthetic, SplitConfig(kcore=2))
    s.train = pd.concat([s.train, s.test.head(1)])
    problems = check_split(s)
    assert any("train has ts >= t1" in p for p in problems)
    assert any("shared by train/test" in p for p in problems)


def test_detects_cold_item_in_test(synthetic):
    s = temporal_split(synthetic, SplitConfig(kcore=2))
    row = s.test.head(1).assign(item=99_999)
    s.test = pd.concat([s.test, row])
    assert any("test contains items absent from train" in p for p in check_split(s))


def test_detects_future_in_mask(synthetic):
    s = temporal_split(synthetic, SplitConfig(kcore=2))
    s.seen_before_val = pd.concat([s.seen_before_val, s.test.head(1)[["user", "item", "ts"]]])
    assert any("seen_before_val" in p for p in check_split(s))


@pytest.mark.parametrize("cfg_name,raw", [("ml100k", "ml-100k"), ("ml1m", "ml-1m")])
def test_real_splits_pass(cfg_name, raw, monkeypatch):
    if not data_available(raw):
        pytest.skip(f"data/raw/{raw} not downloaded")
    monkeypatch.chdir(ROOT)
    split, _ = build_split(load_config(ROOT / "configs" / "data" / f"{cfg_name}.yaml"))
    assert check_split(split) == []
