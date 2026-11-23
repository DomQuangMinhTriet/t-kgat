"""Refit on train + validation (decision 0016 item 4): data, epochs and the structural guards (no torch needed)."""

import ast
import inspect

import numpy as np
import pytest

from tkgat.data.encode import encode, encode_trainval
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.pipeline import refit
from tkgat.pipeline.refit import refit_epochs


@pytest.fixture
def split(synthetic):
    return temporal_split(synthetic, SplitConfig(kcore=2))


def pair_set(enc):
    return set(map(tuple, enc.train.tolist()))


def test_trainval_part_is_train_plus_validation_with_the_same_vocabulary(split):
    base, merged = encode(split), encode_trainval(split)
    assert len(split.val) > 0
    assert np.array_equal(base.user_ids, merged.user_ids) and np.array_equal(base.item_ids, merged.item_ids)
    assert merged.n_users == base.n_users and merged.n_items == base.n_items
    val_pairs = {(u, i) for u, items in encode(split).val.items() for i in items.tolist()}
    assert pair_set(merged) == pair_set(base) | val_pairs and not pair_set(base) & val_pairs
    assert len(merged.train) == len(base.train) + len(split.val) == len(merged.train_ts)


def test_trainval_keeps_test_and_the_test_mask_and_empties_validation(split):
    base, merged = encode(split), encode_trainval(split)
    assert merged.val == {}
    assert merged.test.keys() == base.test.keys()
    assert all(np.array_equal(merged.test[u], base.test[u]) for u in base.test)
    assert (merged.mask_test != base.mask_test).nnz == 0
    # every merged train pair is in the test mask (everything before t2), and no test target is
    train_csr = merged.train_csr()
    assert (train_csr.multiply(merged.mask_test != train_csr)).nnz == 0
    assert all(not merged.mask_test[u, i] for u, items in merged.test.items() for i in items.tolist())


def test_trainval_timestamps_follow_the_merged_pairs_and_move_the_anchor(split):
    base, merged = encode(split), encode_trainval(split)
    assert base.train_ts.max() < split.t1 <= merged.train_ts.max() < split.t2
    order = np.lexsort((merged.train_ts, merged.train[:, 0]))
    assert np.array_equal(order, np.arange(len(order)))  # sorted by (user, ts) like the normal encoding
    assert merged.train_ts.max() > base.train_ts.max()  # T_ref is recomputed on train + validation


def test_a_pair_in_train_and_validation_is_refused(split):
    broken = split.val.copy()
    broken.iloc[0, broken.columns.get_indexer(["user", "item"])] = split.train.iloc[0][["user", "item"]].to_numpy()
    split.val = broken
    with pytest.raises(ValueError, match="both train and val"):
        encode_trainval(split)


def test_refit_epochs_is_the_median_of_the_validation_best_epochs():
    assert refit_epochs([130, 150, 160, 140, 170]) == 150
    assert refit_epochs([100]) == 100
    assert refit_epochs([100, 105]) == 103  # an even number of seeds: the half is rounded up
    for bad in ([], [0, 10], [-5]):
        with pytest.raises(ValueError, match="best epoch"):
            refit_epochs(bad)


def test_the_refit_module_reads_no_split_part_but_the_train_timestamps():
    attributes = {n.attr for n in ast.walk(ast.parse(inspect.getsource(refit))) if isinstance(n, ast.Attribute)}
    assert not attributes & {"test", "val", "mask_test", "mask_val", "seen_before_test", "seen_before_val"}
