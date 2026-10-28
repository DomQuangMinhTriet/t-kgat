"""KGAT model on a 10-node graph: a dense NumPy implementation of the paper's formulas is the reference (CPU only)."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from tkgat.eval.metrics import evaluate
from tkgat.kg.ckg import KGTriples, build_ckg, no_kg
from tkgat.models.kgat import KGAT, KGATConfig

U, NI, E, R = 3, 3, 7, 2  # NI = items; users, items, entities (3 items + 4 attributes), KG relations: 10 nodes
TRAIN = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 0]], dtype=np.int64)
TRAIN_TS = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
# (item, relation, attribute); item 0 reaches attribute 3 through both relations; attribute 6 has no edge at all
TRIPLES = np.array([[0, 0, 3], [0, 1, 3], [0, 0, 4], [1, 0, 4], [1, 1, 5], [2, 1, 5], [2, 0, 3]], dtype=np.int64)
CFG = dict(dim=4, kge_dim=3, layers=(4, 3, 2), mess_dropout=0.0)


def make_ckg(triples=TRIPLES):
    enc = SimpleNamespace(n_users=U, n_items=NI, train=TRAIN, train_ts=TRAIN_TS)
    kg = KGTriples("default", "t1", np.ascontiguousarray(triples), NI, E, ("a", "b"), True, None)
    return build_ckg(enc, kg)


def make_model(ckg=None, seed=0, double=True, **kw):
    torch.manual_seed(seed)
    model = KGAT(ckg or make_ckg(), KGATConfig(**{**CFG, **kw}), device="cpu")
    if double:
        model = model.double()
        model.update_attention()  # the uniform A of construction is float32; this one is float64
    return model


def lrelu(x):
    return np.where(x > 0, x, 0.2 * x)


def reference(model, ckg):
    """Paper formulas on dense arrays, column vectors, ``W_r`` of shape (k, d); eval mode (no dropout)."""
    p = {k: v.detach().double().numpy() for k, v in model.named_parameters()}
    emb = np.concatenate([p["user_embed"], p["entity_embed"]])
    n = len(emb)
    edges = list(zip(ckg.head.tolist(), ckg.relation.tolist(), ckg.tail.tolist(), strict=True))

    pi = []
    for h, r, t in edges:
        w = p["trans_W"][r].T  # eq. 1: e^r = W_r e
        pi.append((w @ emb[t]) @ np.tanh(w @ emb[h] + p["relation_embed"][r]))  # eq. 4
    pi = np.array(pi)
    att = np.zeros((n, n))
    for h in range(n):  # eq. 5: softmax over every edge with head h
        mine = [i for i, e in enumerate(edges) if e[0] == h]
        if mine:
            x = np.exp(pi[mine] - pi[mine].max())
            for i, v in zip(mine, x / x.sum(), strict=True):
                att[edges[i][0], edges[i][2]] += v

    layers, ego = [emb], emb
    for k in range(len(model.aggregators)):
        w1, b1 = p[f"aggregators.{k}.w1"].T, p[f"aggregators.{k}.b1"][0]  # paper: W1 is (d_out, d_in)
        w2, b2 = p[f"aggregators.{k}.w2"].T, p[f"aggregators.{k}.b2"][0]
        nxt = []
        for v in range(n):
            e_n = att[v] @ ego  # eq. 10
            nxt.append(lrelu(w1 @ (ego[v] + e_n) + b1) + lrelu(w2 @ (ego[v] * e_n) + b2))  # eqs. 8-9
        ego = np.array(nxt)
        layers.append(ego / np.linalg.norm(ego, axis=1, keepdims=True))
    return att, layers, np.concatenate(layers, axis=1)


def test_the_graph_has_ten_nodes_and_an_isolated_entity():
    m = make_model()
    assert m.n_nodes == 10 and int((m._degree == 0).sum()) == 1 and m.n_rel == 6


def test_attention_matches_the_dense_reference():
    ckg = make_ckg()
    m = make_model(ckg)
    att, _, _ = reference(m, ckg)
    assert np.allclose(m.A.to_dense().numpy(), att, atol=1e-12)


def test_attention_of_every_head_node_sums_to_one():
    m = make_model()
    rows = m.A.to_dense().sum(1).numpy()
    has_edges = (m._degree > 0).numpy()
    assert np.allclose(rows[has_edges], 1.0) and (rows[~has_edges] == 0).all()
    assert (m.A.to_dense() >= 0).all()


def test_attention_starts_uniform_over_the_edges_of_a_head():
    m = make_model(double=False)
    dense = m.A.to_dense().numpy()
    ckg = make_ckg()
    deg = np.bincount(ckg.head, minlength=m.n_nodes)
    for h, _, t in zip(ckg.head, ckg.relation, ckg.tail, strict=True):
        multi = int(((ckg.head == h) & (ckg.tail == t)).sum())  # item 0 -> attribute 3 has two edges
        assert dense[h, t] == pytest.approx(multi / deg[h], rel=1e-6)


def test_attention_is_the_same_in_small_chunks():
    ckg = make_ckg()
    big, small = make_model(ckg), make_model(ckg, attention_chunk=2)
    assert torch.equal(big.A.to_dense(), small.A.to_dense())


def test_propagation_layers_and_output_match_the_dense_reference():
    ckg = make_ckg()
    m = make_model(ckg)
    _, layers, star = reference(m, ckg)
    got = m.eval_embeddings().numpy()
    assert got.shape == (10, 4 + 4 + 3 + 2) == star.shape
    assert np.allclose(got, star, atol=1e-10)
    start = 4  # layers >= 1 are unit vectors, the input embedding is not normalised
    for layer in layers[1:]:
        width = layer.shape[1]
        assert np.allclose(np.linalg.norm(got[:, start : start + width], axis=1), 1.0)
        start += width
    assert not np.allclose(np.linalg.norm(got[:, :4], axis=1), 1.0)


def test_scores_are_inner_products_of_the_concatenated_vectors():
    ckg = make_ckg()
    m = make_model(ckg)
    _, _, star = reference(m, ckg)
    users = np.array([0, 2])
    want = star[users] @ star[U : U + NI].T
    assert np.allclose(m.score(users).numpy(), want, atol=1e-10)
    assert np.allclose(m.scorer()(np.array([1])).numpy(), star[[1]] @ star[U : U + NI].T, atol=1e-10)


def test_scorer_works_with_evaluate_and_ignores_dropout():
    m = make_model(double=False, mess_dropout=0.5)
    m.train()
    score = m.scorer()
    assert m.training and torch.equal(score(np.array([0, 1])), score(np.array([0, 1])))
    res = evaluate(score, {0: np.array([2]), 1: np.array([0])}, sp.csr_matrix((U, NI)), ks=(2,))
    assert res["n_users"] == 2 and 0.0 <= res["recall@2"] <= 1.0


def test_message_dropout_acts_in_training_mode_only():
    m = make_model(double=False, mess_dropout=0.5)
    m.train()
    torch.manual_seed(1)
    a, b = m.propagate(), m.propagate()
    assert not torch.equal(a, b)
    m.eval()
    assert torch.equal(m.propagate(), m.propagate())


def test_cf_loss_matches_the_hand_computation():
    ckg = make_ckg()
    m = make_model(ckg, l2=0.01)
    _, _, star = reference(m, ckg)
    u, i, j = np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([2, 0, 1])
    x = (star[u] * star[U + i]).sum(1) - (star[u] * star[U + j]).sum(1)
    reg = 0.5 * np.mean((star[u] ** 2).sum(1) + (star[U + i] ** 2).sum(1) + (star[U + j] ** 2).sum(1))
    want = np.mean(np.log1p(np.exp(-x))) + 0.01 * reg
    m.eval()
    got = m.cf_loss(torch.tensor(u), torch.tensor(i), torch.tensor(j)).item()
    assert got == pytest.approx(want, rel=1e-9)


def test_kg_loss_matches_the_hand_computation():
    ckg = make_ckg()
    m = make_model(ckg, l2_kg=0.01)
    p = {k: v.detach().numpy() for k, v in m.named_parameters()}
    emb = np.concatenate([p["user_embed"], p["entity_embed"]])
    h, r, t, tn = np.array([3, 4, 5]), np.array([1, 2, 5]), np.array([6, 9, 7]), np.array([8, 3, 9])
    lp, ln, reg = [], [], []
    for a, rel, b, c in zip(h, r, t, tn, strict=True):
        w = p["trans_W"][rel].T
        eh, et, en, er = w @ emb[a], w @ emb[b], w @ emb[c], p["relation_embed"][rel]
        lp.append(((eh + er - et) ** 2).sum())
        ln.append(((eh + er - en) ** 2).sum())
        reg.append((eh**2).sum() + (er**2).sum() + (et**2).sum() + (en**2).sum())
    want = np.mean(np.log1p(np.exp(np.array(lp) - np.array(ln)))) + 0.01 * 0.5 * np.mean(reg)
    got = m.kg_loss(*(torch.tensor(x) for x in (h, r, t, tn))).item()
    assert got == pytest.approx(want, rel=1e-9)


def test_one_optimiser_step_lowers_each_loss():
    ckg = make_ckg()
    batch_cf = (torch.tensor([0, 1, 2, 0]), torch.tensor([0, 1, 0, 1]), torch.tensor([2, 0, 1, 2]))
    batch_kg = tuple(torch.as_tensor(x[:6]) for x in (ckg.head, ckg.relation, ckg.tail, np.roll(ckg.tail, 3)))
    for loss_fn, params in (
        (lambda m: m.cf_loss(*batch_cf), "cf_parameters"),
        (lambda m: m.kg_loss(*batch_kg), "kg_parameters"),
    ):
        m = make_model(ckg, double=False, l2=1e-4, l2_kg=1e-4)
        opt = torch.optim.Adam(getattr(m, params)(), lr=0.05)
        before = loss_fn(m)
        opt.zero_grad()
        before.backward()
        opt.step()
        assert loss_fn(m).item() < before.item()


def test_each_optimiser_updates_only_its_own_parameters():
    m = make_model(double=False)

    def ids(params):
        return {id(p) for p in params}

    cf, kg = ids(m.cf_parameters()), ids(m.kg_parameters())
    assert ids([m.user_embed, m.entity_embed]) <= cf & kg
    assert ids(m.aggregators.parameters()) <= cf and not (ids(m.aggregators.parameters()) & kg)
    assert ids([m.relation_embed, m.trans_W]) <= kg and not (ids([m.relation_embed, m.trans_W]) & cf)
    assert cf | kg == ids(m.parameters())
    # the attention gradient never reaches the CF loss: A is a constant buffer without grad
    assert not m.A.requires_grad
    m.cf_loss(torch.tensor([0]), torch.tensor([0]), torch.tensor([1])).backward()
    assert m.relation_embed.grad is None and m.trans_W.grad is None


def test_gamma_zero_gives_the_tensors_of_the_model_without_gamma():
    ckg = make_ckg()
    plain = make_model(ckg, seed=3)  # gamma not passed
    zero = make_model(ckg, seed=3, gamma=0.0)
    zero.edge_ts.copy_(torch.linspace(1.0, 9e8, len(zero.edge_ts)))  # time inputs must not matter at gamma = 0
    zero.delta_r.fill_(1.0)
    zero.update_attention()
    assert torch.equal(plain.A.to_dense(), zero.A.to_dense())
    assert torch.equal(plain.eval_embeddings(), zero.eval_embeddings())
    nan_ts = make_model(ckg, seed=3)
    assert torch.isnan(nan_ts.edge_ts).any()  # the KG edges carry NaN; gamma = 0 must not touch them
    assert torch.isfinite(nan_ts.eval_embeddings()).all()


def test_the_model_copies_the_graph_arrays():
    ckg = make_ckg()
    head, ts = ckg.head.copy(), ckg.edge_ts.copy()
    m = make_model(ckg, double=False)
    m.head.zero_()
    m.edge_ts.fill_(1.0)
    assert np.array_equal(ckg.head, head) and np.array_equal(ckg.edge_ts, ts, equal_nan=True)


def test_the_graph_can_be_replaced_and_attention_follows_it():
    ckg1, ckg2 = make_ckg(), make_ckg(TRIPLES[:4])
    m = make_model(ckg1)
    before = m.A.to_dense().clone()
    m.set_graph(ckg2)
    assert m.A.to_dense().sum().item() == pytest.approx(float((m._degree > 0).sum()))  # uniform again
    m.update_attention()
    att, _, star = reference(m, ckg2)
    assert np.allclose(m.A.to_dense().numpy(), att, atol=1e-12) and not torch.equal(m.A.to_dense(), before)
    assert np.allclose(m.eval_embeddings().numpy(), star, atol=1e-10)


def test_a_graph_with_other_id_spaces_is_refused():
    enc = SimpleNamespace(n_users=U + 1, n_items=NI, train=TRAIN, train_ts=TRAIN_TS)
    other = build_ckg(enc, KGTriples("default", "t1", TRIPLES, NI, E, ("a", "b"), True, None))
    with pytest.raises(ValueError, match="id spaces"):
        make_model().set_graph(other)


def test_state_dict_holds_parameters_only_and_loads_on_another_graph():
    a = make_model(make_ckg(), seed=1, double=False)
    b = make_model(make_ckg(TRIPLES[:4]), seed=2, double=False)
    assert set(a.state_dict()) == {name for name, _ in a.named_parameters()}
    b.load_state_dict(a.state_dict())
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_a_graph_without_kg_has_two_relations_and_attention_over_interactions():
    enc = SimpleNamespace(n_users=U, n_items=NI, train=TRAIN, train_ts=TRAIN_TS)
    m = make_model(build_ckg(enc, no_kg(enc)))
    assert m.n_rel == 2 and m.trans_W.shape == (2, 4, 3)
    assert np.allclose(m.A.to_dense().sum(1).numpy()[: U + NI], 1.0)
