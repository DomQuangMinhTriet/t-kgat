"""GPU support: negatives, device-side evaluation and BPR training on any device."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from tkgat.data.encode import encode
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.device import resolve_device
from tkgat.eval.metrics import evaluate, recall_ndcg_at_k
from tkgat.train.bpr import BPRConfig, NegativeSampler, fit_bpr

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _reference_evaluate(scores, truth, mask, ks=(10, 20)):
    """The pre-GPU NumPy implementation, kept here as the oracle."""
    users = np.array(sorted(truth))
    kmax = min(max(ks), scores.shape[1])
    out = {f"{m}@{k}": [] for k in ks for m in ("recall", "ndcg")}
    s = scores[users].astype(np.float64).copy()
    s[mask[users].toarray().astype(bool)] = -np.inf
    top = np.argpartition(-s, kmax - 1, axis=1)[:, :kmax]
    order = np.take_along_axis(s, top, axis=1).argsort(axis=1)[:, ::-1]
    top = np.take_along_axis(top, order, axis=1)
    for j, u in enumerate(users):
        for k in ks:
            r, n = recall_ndcg_at_k(top[j], truth[int(u)], k)
            out[f"recall@{k}"].append(r)
            out[f"ndcg@{k}"].append(n)
    return {k: float(np.nanmean(v)) for k, v in out.items()}


@pytest.mark.parametrize("device", DEVICES)
def test_negative_sampler_never_returns_a_train_positive(device):
    rng = np.random.default_rng(0)
    n_users, n_items = 50, 30
    pairs = np.array([(u, i) for u in range(n_users) for i in rng.choice(n_items, 25, replace=False)])
    train = torch.as_tensor(pairs, device=device)
    gen = torch.Generator(device=device).manual_seed(0)
    sampler = NegativeSampler(train, n_items, gen)
    users = train[:, 0].repeat(20)
    neg = sampler(users).cpu().numpy()
    pos = {(int(u), int(i)) for u, i in pairs}
    assert not any((int(u), int(j)) in pos for u, j in zip(users.cpu().numpy(), neg, strict=True))


@pytest.mark.parametrize("device", DEVICES)
def test_device_evaluation_matches_numpy_reference(device):
    rng = np.random.default_rng(1)
    n_users, n_items = 40, 60
    scores = rng.normal(size=(n_users, n_items))
    mask = sp.csr_matrix((rng.random((n_users, n_items)) < 0.2).astype(np.float32))
    truth = {u: rng.choice(n_items, 3, replace=False) for u in range(0, n_users, 2)}
    t = torch.as_tensor(scores, device=device)
    got = evaluate(lambda b: t[torch.as_tensor(b, device=device)], truth, mask)
    ref = _reference_evaluate(scores, truth, mask)
    for k, v in ref.items():
        assert got[k] == pytest.approx(v, abs=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_bpr_runs_and_is_reproducible_on_device(synthetic, device):
    data = encode(temporal_split(synthetic, SplitConfig(kcore=2)))
    cfg = BPRConfig(dim=8, max_epochs=10, eval_every=5, patience=2, batch_size=256, device=device)
    a, b = fit_bpr(data, cfg, seed=7), fit_bpr(data, cfg, seed=7)
    assert a.device.startswith(device)
    assert a.best_val == b.best_val and a.best_epoch == b.best_epoch
    assert next(a.model.parameters()).device.type == device


def test_cuda_request_without_gpu_fails_loudly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


@pytest.mark.parametrize("device", DEVICES)
def test_k_larger_than_the_number_of_items(device):
    """Fewer items than K: every item is ranked, no error, and the @K values match the NumPy oracle."""
    rng = np.random.default_rng(2)
    n_users, n_items = 12, 5
    scores = rng.normal(size=(n_users, n_items))
    mask = sp.csr_matrix((rng.random((n_users, n_items)) < 0.2).astype(np.float32))
    # as in the real splits, ground-truth items are never in the mask (masked items tie at -inf, and
    # their order among themselves is implementation-defined)
    dense = mask.toarray().astype(bool)
    truth = {}
    for u in range(n_users):
        unseen = np.flatnonzero(~dense[u])
        truth[u] = rng.choice(unseen, min(2, len(unseen)), replace=False)
    t = torch.as_tensor(scores, device=device)
    got = evaluate(lambda b: t[torch.as_tensor(b, device=device)], truth, mask, ks=(10,))
    ref = _reference_evaluate(scores, truth, mask, ks=(10,))
    for k, v in ref.items():
        assert got[k] == pytest.approx(v, abs=1e-12)


def test_k_larger_than_the_number_of_items_by_hand():
    # items ranked 2, 0, 4, 1, 3; truth {0, 3}: with K = 10 > 5 items both are found
    scores = np.array([[0.8, 0.2, 0.9, 0.1, 0.5]])
    res = evaluate(lambda users: scores, {0: np.array([0, 3])}, sp.csr_matrix((1, 5), dtype=np.float32), ks=(10,))
    assert res["recall@10"] == 1.0
    dcg = 1 / np.log2(3) + 1 / np.log2(6)  # ranks 2 and 5
    idcg = 1 / np.log2(2) + 1 / np.log2(3)  # min(K, |truth|) = 2
    assert res["ndcg@10"] == pytest.approx(dcg / idcg)
