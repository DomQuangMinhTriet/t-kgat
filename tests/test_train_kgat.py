"""KGAT training on a small synthetic split (CPU): it runs, it is deterministic and it never needs the test split."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tkgat.train.kgat as trainer
from tkgat.data.encode import encode
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.kg.ckg import KGTriples, build_ckg
from tkgat.models.kgat import KGAT, KGATConfig
from tkgat.pipeline.build import load_saved, save_split
from tkgat.train.kgat import KGATFitConfig, KGNegativeSampler, fit_kgat, split_params

N_ATTR = 6
MODEL = KGATConfig(dim=8, kge_dim=8, layers=(8, 4), mess_dropout=0.1)
FIT = KGATFitConfig(lr=0.01, batch_size=64, batch_size_kg=64, max_epochs=3, eval_every=1, patience=3, device="cpu")
TIMING = ("seconds", "sec_cf", "sec_kg", "sec_attention", "sec_eval")


@pytest.fixture
def world(synthetic, tmp_path):
    split = temporal_split(synthetic, SplitConfig(kcore=2))
    cfg = {"name": "synthetic", "processed_dir": str(tmp_path / "processed")}
    save_split(split, cfg, {})
    return cfg, encode(split)


def make_ckg(enc):
    n = enc.n_items
    items = np.arange(n)
    triples = np.concatenate(
        [
            np.stack([items, np.zeros(n, dtype=int), n + items % N_ATTR], axis=1),
            np.stack([items, np.ones(n, dtype=int), n + (items * 5 + 1) % N_ATTR], axis=1),
        ]
    )
    triples = np.unique(triples, axis=0).astype(np.int64)
    kg = KGTriples("default", "t1", triples, n, n + N_ATTR, ("a", "b"), True, None)
    return build_ckg(enc, kg)


def stable(history):
    """The history without the wall-clock columns, which differ from run to run."""
    return [{k: v for k, v in row.items() if k not in TIMING} for row in history]


def same_state(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_one_epoch_runs(world):
    _, enc = world
    assert len(enc.val) > 0
    res = fit_kgat(enc, make_ckg(enc), MODEL, replace(FIT, max_epochs=1), seed=0)
    row = res.history[0]
    assert len(res.history) == 1 and res.best_epoch == 1 and row["epoch"] == 1
    assert np.isfinite([row["loss_cf"], row["loss_kg"]]).all() and row["loss_cf"] > 0 and row["loss_kg"] > 0
    assert 0.0 <= row["recall@20"] <= 1.0 and set(res.best_val) == set(trainer.METRICS)
    assert row["seconds"] >= row["sec_cf"] + row["sec_kg"] + row["sec_attention"] - 1e-9 and row["sec_eval"] >= 0
    assert all(t.device.type == "cpu" for t in res.best_state.values()) and res.peak_memory == 0
    assert res.device == "cpu" and not res.model.training
    assert same_state(res.best_state, {k: v.cpu() for k, v in res.model.state_dict().items()})


def test_same_seed_gives_the_same_history_and_weights(world):
    _, enc = world
    ckg = make_ckg(enc)
    a, b = (fit_kgat(enc, ckg, MODEL, FIT, seed=7) for _ in range(2))
    assert stable(a.history) == stable(b.history) and len(a.history) == 3
    assert same_state(a.best_state, b.best_state) and torch.equal(a.model.A.to_dense(), b.model.A.to_dense())
    other = fit_kgat(enc, ckg, MODEL, FIT, seed=8)
    assert stable(other.history) != stable(a.history)


def test_training_never_touches_the_test_split(world, monkeypatch):
    cfg, enc = world
    ckg = make_ckg(enc)
    reference = fit_kgat(enc, ckg, MODEL, FIT, seed=1)

    # (a) an object that has no test attributes at all: any access raises AttributeError
    blind = SimpleNamespace(n_items=enc.n_items, train=enc.train, val=enc.val, mask_val=enc.mask_val)
    res = fit_kgat(blind, ckg, MODEL, FIT, seed=1)
    assert stable(res.history) == stable(reference.history) and same_state(res.best_state, reference.best_state)

    # (b) the test files do not exist on disk: they are neither opened by the loader nor needed by the guard
    for name in ("test.csv", "seen_before_test.csv"):
        (Path(cfg["processed_dir"]) / name).unlink()
    split, _ = load_saved(cfg, with_test=False)
    enc_val_only = encode(split)
    assert enc_val_only.test == {} and enc_val_only.mask_test.nnz == 0
    res = fit_kgat(enc_val_only, make_ckg(enc_val_only), MODEL, FIT, seed=1)
    assert stable(res.history) == stable(reference.history) and same_state(res.best_state, reference.best_state)

    # (c) whatever the test fields hold, the run is the same; evaluate never sees them
    seen = []
    real = trainer.evaluate

    def spy(score, truth, mask, **kw):
        seen.append((truth, mask))
        return real(score, truth, mask, **kw)

    monkeypatch.setattr(trainer, "evaluate", spy)
    junk = SimpleNamespace(**{**vars(blind), "test": {0: np.arange(3)}, "mask_test": enc.mask_val})
    fit_kgat(junk, ckg, MODEL, FIT, seed=1)
    assert seen and all(t is junk.val and m is junk.mask_val for t, m in seen)


def test_load_saved_without_test_reads_the_same_train_and_val(world):
    cfg, enc = world
    for name in ("test.csv", "seen_before_test.csv"):
        (Path(cfg["processed_dir"]) / name).unlink()
    split, meta = load_saved(cfg, with_test=False)
    got = encode(split)
    assert np.array_equal(got.train, enc.train) and np.array_equal(got.train_ts, enc.train_ts)
    assert got.val.keys() == enc.val.keys() and (got.mask_val != enc.mask_val).nnz == 0
    assert len(split.test) == 0 and len(split.seen_before_test) == 0 and "sha256" in meta
    with pytest.raises(FileNotFoundError):
        load_saved(cfg)  # the default still needs every part


def test_the_first_cf_phase_uses_uniform_attention_and_it_is_updated_every_epoch(world, monkeypatch):
    _, enc = world
    ckg = make_ckg(enc)
    first, calls = [], []
    real_cf, real_update = KGAT.cf_loss, KGAT.update_attention

    def cf_loss(self, *args):
        if not first:
            first.append(self.A.to_dense().clone())
        return real_cf(self, *args)

    def update(self):
        calls.append(1)
        return real_update(self)

    monkeypatch.setattr(KGAT, "cf_loss", cf_loss)
    monkeypatch.setattr(KGAT, "update_attention", update)
    res = fit_kgat(enc, ckg, MODEL, FIT, seed=0)
    uniform = KGAT(ckg, MODEL, device="cpu")
    uniform.reset_attention()
    assert torch.allclose(first[0], uniform.A.to_dense())
    assert len(calls) == len(res.history) + 1  # after every epoch, and once more for the best state
    assert not torch.allclose(res.model.A.to_dense(), uniform.A.to_dense())


def test_patience_counts_evaluations_and_the_best_state_is_restored(world, monkeypatch):
    _, enc = world
    values = iter([0.10, 0.30, 0.20, 0.25, 0.28, 0.29])  # best at the 2nd evaluation, then no improvement
    metrics = dict.fromkeys(trainer.METRICS, 0.0)

    def fake(score, truth, mask, **kw):
        return {**metrics, "recall@20": next(values), "n_users": len(truth)}

    monkeypatch.setattr(trainer, "evaluate", fake)
    cfg = replace(FIT, max_epochs=6, patience=2)
    res = fit_kgat(enc, make_ckg(enc), MODEL, cfg, seed=0)
    assert [r["epoch"] for r in res.history] == [1, 2, 3, 4]  # stops after two evaluations without improvement
    assert res.best_epoch == 2 and res.best_val["recall@20"] == 0.30
    assert same_state(res.best_state, {k: v.cpu() for k, v in res.model.state_dict().items()})


def test_no_evaluation_is_an_error(world):
    _, enc = world
    cfg = replace(FIT, max_epochs=2, eval_every=5)
    with pytest.raises(ValueError, match="eval_every|max_epochs"):
        fit_kgat(enc, make_ckg(enc), MODEL, cfg, seed=0)


def edges_of(ckg):
    return torch.tensor(np.stack([ckg.head, ckg.relation, ckg.tail], axis=1))


def test_corrupted_tails_are_never_true_edges_and_never_isolated_nodes(world):
    _, enc = world
    ckg = make_ckg(enc)
    edges = edges_of(ckg)
    sampler = KGNegativeSampler(edges, ckg.n_nodes, ckg.n_rel, torch.Generator().manual_seed(0))
    h, r = edges[:, 0].repeat(5), edges[:, 1].repeat(5)
    neg = sampler(h, r)
    assert not sampler.is_edge(h, r, neg).any()
    degree = np.bincount(ckg.head, minlength=ckg.n_nodes)
    assert (degree[neg.numpy()] > 0).all()


def test_isolated_nodes_are_never_drawn_as_corrupted_tails():
    # nodes 0..3 have edges, node 4 has none
    edges = torch.tensor([[0, 0, 1], [1, 1, 0], [2, 0, 3], [3, 1, 2]])
    sampler = KGNegativeSampler(edges, 5, 2, torch.Generator().manual_seed(0))
    neg = sampler(edges[:, 0].repeat(500), edges[:, 1].repeat(500))
    assert 4 not in set(neg.tolist()) and set(neg.tolist()) <= {0, 1, 2, 3}


def test_a_head_whose_every_candidate_is_a_true_tail_raises():
    edges = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 0]])  # nodes 0 and 1 are active; (0, r0) reaches both
    sampler = KGNegativeSampler(edges, 2, 1, torch.Generator().manual_seed(0))
    with pytest.raises(RuntimeError, match="no corrupted tail"):
        sampler(torch.tensor([0]), torch.tensor([0]))


def test_split_params_separates_model_and_trainer_settings():
    model, fit = split_params({"lr": 0.1, "l2": 1e-3, "layers": [8, 4], "patience": 4, "mess_dropout": 0.2}, "cpu")
    assert (model.l2, model.layers, model.mess_dropout) == (1e-3, (8, 4), 0.2)
    assert (fit.lr, fit.patience, fit.device) == (0.1, 4, "cpu")
    with pytest.raises(ValueError, match="unknown"):
        split_params({"learning_rate": 0.1})
