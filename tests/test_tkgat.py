"""T-KGAT Stage 1 (decision 0009): Delta-t in the attention, its anchor, the permutation control, the registry rows.

Everything runs on the CPU with tiny graphs; nothing here trains on real data or touches the GPU.
"""

import dataclasses
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from tkgat.data.encode import encode
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.kg import ckg as ckg_module
from tkgat.kg.ckg import INTERACT, KGTriples, build_ckg, inverse_relation_id, no_kg, permute_train_ts
from tkgat.models import kgat as model_module
from tkgat.models.kgat import KGAT, KGATConfig
from tkgat.pipeline.build import load_saved, save_split
from tkgat.train import kgat as trainer_module
from tkgat.train.kgat import split_params

from .conftest import ROOT
from .test_kgat import CFG, NI, TRAIN, TRAIN_TS, TRIPLES, E, U, make_ckg, make_model, reference
from .test_run_kgat import FIXED, GRID, RUN_NAME, load_run_script, write_kg

RELATIONS = ("a", "b")


def timed(model):
    """Boolean mask of the edges that carry a timestamp (Interact and its inverse)."""
    return (model.delta_r[model.relation] > 0).numpy()


def reference_attention(model, ckg, gamma):
    """The paper's attention on dense arrays, with ``- gamma * delta_r * dt_norm`` added to the logits by hand."""
    p = {k: v.detach().double().numpy() for k, v in model.named_parameters()}
    emb = np.concatenate([p["user_embed"], p["entity_embed"]])
    n, n_rel_kg = len(emb), ckg.n_relations
    pi = []
    for h, r, t, ts in zip(ckg.head, ckg.relation, ckg.tail, ckg.edge_ts, strict=True):
        w = p["trans_W"][r].T
        score = (w @ emb[t]) @ np.tanh(w @ emb[h] + p["relation_embed"][r])
        if r in (INTERACT, inverse_relation_id(INTERACT, n_rel_kg)):
            score -= gamma * max(0.0, ckg.t_ref - ts) / ckg.dt_den
        pi.append(score)
    pi = np.array(pi)
    att = np.zeros((n, n))
    for h in range(n):
        mine = np.flatnonzero(ckg.head == h)
        if len(mine):
            x = np.exp(pi[mine] - pi[mine].max())
            for i, v in zip(mine, x / x.sum(), strict=True):
                att[ckg.head[i], ckg.tail[i]] += v
    return att


def graph_with_ts(ts):
    enc = SimpleNamespace(n_users=U, n_items=NI, train=TRAIN, train_ts=np.asarray(ts, dtype=np.float64))
    kg = KGTriples("default", "t1", np.ascontiguousarray(TRIPLES), NI, E, RELATIONS, True, None)
    return build_ckg(enc, kg)


# --- the formula --------------------------------------------------------------------------------------------------


def test_gamma_zero_is_kgat():
    ckg = make_ckg()
    plain = make_model(ckg, seed=3)  # no time at all
    zero = make_model(ckg, seed=3, gamma=0.0)
    zero.edge_ts.copy_(torch.linspace(1.0, 9e8, len(zero.edge_ts)))  # whatever the time inputs hold ...
    zero.delta_r.fill_(1.0)
    zero.update_attention()
    assert torch.equal(plain.A.to_dense(), zero.A.to_dense())  # ... A and e* are the same, element by element
    assert torch.equal(plain.eval_embeddings(), zero.eval_embeddings())
    att, _, star = reference(zero, ckg)  # and they are the paper's KGAT
    assert np.allclose(zero.A.to_dense().numpy(), att, atol=1e-12)
    assert np.allclose(zero.eval_embeddings().numpy(), star, atol=1e-10)
    assert not torch.equal(plain.A.to_dense(), make_model(ckg, seed=3, gamma=2.0).A.to_dense())  # gamma does act


def test_zero_dt_is_kgat():
    # (a) every train timestamp equal: t_ref - min = 0, so dt_norm = 0 everywhere
    ckg = graph_with_ts([700.0] * len(TRAIN))
    assert ckg.dt_den == 0.0
    plain, with_gamma = make_model(ckg, seed=4), make_model(ckg, seed=4, gamma=5.0)
    assert torch.equal(plain.A.to_dense(), with_gamma.A.to_dense())
    assert torch.equal(plain.eval_embeddings(), with_gamma.eval_embeddings())
    # (b) real timestamps, but dt set to 0 on every edge (ts = t_ref)
    ckg = make_ckg()
    plain, with_gamma = make_model(ckg, seed=4), make_model(ckg, seed=4, gamma=5.0)
    assert not torch.equal(plain.A.to_dense(), with_gamma.A.to_dense())  # with dt > 0 it differs ...
    with_gamma.edge_ts.copy_(torch.where(torch.isnan(with_gamma.edge_ts), with_gamma.edge_ts, ckg.t_ref))
    with_gamma.update_attention()
    assert torch.equal(plain.A.to_dense(), with_gamma.A.to_dense())  # ... and with dt = 0 it does not
    assert torch.equal(plain.eval_embeddings(), with_gamma.eval_embeddings())


@pytest.mark.parametrize("gamma", [0.5, 1.0, 3.0, 10.0])
def test_attention_matches_the_dense_reference_with_the_time_term(gamma):
    ckg = make_ckg()
    model = make_model(ckg, seed=2, gamma=gamma)
    assert ckg.t_ref == TRAIN_TS.max() and ckg.dt_den == TRAIN_TS.max() - TRAIN_TS.min()
    assert np.allclose(model.A.to_dense().numpy(), reference_attention(model, ckg, gamma), atol=1e-12)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 3.0, 10.0])
def test_an_older_edge_never_has_more_attention_when_pi_is_equal(gamma):
    model = make_model(make_ckg(), gamma=gamma)
    model._raw_scores = lambda: torch.zeros(len(model.head), dtype=torch.float64)  # equal pi on every edge
    weights = model._segment_softmax(model._edge_scores()).numpy()
    ts, head, mask = model.edge_ts.numpy(), model.head.numpy(), timed(model)
    checked = 0
    for h in np.unique(head):
        here = head == h
        older_first = np.flatnonzero(here & mask)[np.argsort(ts[here & mask], kind="stable")]
        w = weights[older_first]
        assert (np.diff(w) >= 0).all()  # the older the edge, the smaller (never larger) the weight
        if gamma > 0 and len(older_first) > 1 and len(set(ts[older_first])) > 1:
            assert w[0] < w[-1]
            checked += 1
        untimed = weights[here & ~mask]
        if len(untimed) and len(w):
            assert untimed.min() >= w.max()  # a KG edge is never penalised against an interaction
    assert gamma == 0 or checked  # the toy graph has an item with two interactions of different age


def test_delta_r_is_one_for_interact_and_its_inverse_only():
    model = make_model()
    expect = np.zeros(model.n_rel)
    expect[[INTERACT, inverse_relation_id(INTERACT, model.n_relations)]] = 1.0
    assert np.array_equal(model.delta_r.numpy(), expect)


def test_nan_edge_ts_of_kg_edges_never_reaches_the_output():
    ckg = make_ckg()
    model = make_model(ckg, seed=5, gamma=4.0)
    assert torch.isnan(model.edge_ts).any() and torch.isnan(model.edge_ts[~torch.tensor(timed(model))]).all()
    penalty = model._time_penalty(torch.float64)
    assert torch.isfinite(penalty).all() and (penalty[~torch.tensor(timed(model))] == 0).all()
    assert torch.isfinite(model.A.to_dense()).all() and torch.isfinite(model.eval_embeddings()).all()
    # with delta_r = 0 on every relation the time term vanishes and the model is KGAT again
    model.delta_r.zero_()
    model.update_attention()
    assert torch.equal(model.A.to_dense(), make_model(ckg, seed=5).A.to_dense())


def test_a_timed_edge_without_a_timestamp_is_refused_but_plain_kgat_ignores_it():
    ckg = make_ckg()
    ts = ckg.edge_ts.copy()
    ts[np.flatnonzero(~np.isnan(ts))[0]] = np.nan
    broken = dataclasses.replace(ckg, edge_ts=ts)
    with pytest.raises(ValueError, match="no timestamp"):
        KGAT(broken, KGATConfig(**CFG, gamma=1.0), device="cpu")
    KGAT(broken, KGATConfig(**CFG), device="cpu")  # gamma = 0 never reads edge_ts


def test_gamma_must_not_be_negative_and_needs_the_anchor():
    with pytest.raises(ValueError, match="gamma"):
        KGATConfig(gamma=-0.1)
    with pytest.raises(ValueError, match="gamma"):
        KGATConfig(gamma=float("nan"))
    no_anchor = dataclasses.replace(make_ckg(), t_ref=None, dt_den=None)
    with pytest.raises(ValueError, match="t_ref"):
        KGAT(no_anchor, KGATConfig(**CFG, gamma=1.0), device="cpu")


def test_edge_ts_must_stay_float64_when_gamma_is_positive():
    model = make_model(double=False, gamma=1.0).float()
    with pytest.raises(TypeError, match="float64"):
        model.update_attention()


def test_the_time_term_gets_no_gradient_and_gamma_is_a_constant():
    model = make_model(gamma=2.0)
    assert not model.A.requires_grad and not any("gamma" in name for name, _ in model.named_parameters())


# --- where the timestamps come from ---------------------------------------------------------------------------------


@pytest.fixture
def split_enc(synthetic):
    split = temporal_split(synthetic, SplitConfig(kcore=2))
    return split, encode(split)


def test_edge_ts_comes_only_from_train_ts(split_enc):
    split, enc = split_enc
    ckg = build_ckg(enc, no_kg(enc))
    finite = ckg.edge_ts[~np.isnan(ckg.edge_ts)]
    assert np.array_equal(np.sort(finite), np.sort(np.concatenate([enc.train_ts, enc.train_ts]).astype(float)))
    assert finite.max() < split.t1 <= min(split.val["ts"].min(), split.test["ts"].min())  # val and test are later
    assert ckg.t_ref == enc.train_ts.max() and ckg.dt_den == enc.train_ts.max() - enc.train_ts.min()
    # an encoded split that has only the train fields gives the same graph: nothing else is read
    blind = SimpleNamespace(n_users=enc.n_users, n_items=enc.n_items, train=enc.train, train_ts=enc.train_ts)
    other = build_ckg(blind, no_kg(enc))
    assert np.array_equal(other.edge_ts, ckg.edge_ts, equal_nan=True) and other.t_ref == ckg.t_ref


def test_the_graph_at_t2_has_the_same_edge_ts_and_anchor(split_enc):
    _, enc = split_enc
    t1 = build_ckg(enc, no_kg(enc, "t1"))
    t2 = build_ckg(enc, no_kg(enc, "t2"), known_tails=np.array([], dtype=np.int64))
    assert np.array_equal(t1.edge_ts, t2.edge_ts, equal_nan=True)
    assert (t1.t_ref, t1.dt_den) == (t2.t_ref, t2.dt_den)
    model = KGAT(t1, KGATConfig(**CFG, gamma=1.0), device="cpu")
    model.set_graph(t2)
    assert (model._t_ref, model._dt_den) == (t1.t_ref, t1.dt_den)  # the same anchor at val and at test


def test_the_permutation_keeps_the_multiset_and_changes_only_train_ts(split_enc):
    _, enc = split_enc
    original = enc.train_ts.copy()
    shuffled = permute_train_ts(enc, seed=2019)
    assert np.array_equal(np.sort(shuffled.train_ts), np.sort(original))  # the multiset of timestamps is the same
    assert not np.array_equal(shuffled.train_ts, original) and len(shuffled.train_ts) == len(original)
    assert np.array_equal(enc.train_ts, original)  # the input is not modified
    for name in ("train", "val", "test", "mask_val", "mask_test", "user_ids", "item_ids", "n_users", "n_items"):
        assert getattr(shuffled, name) is getattr(enc, name)  # only train_ts is a new object
    again = permute_train_ts(enc, seed=2019)
    assert np.array_equal(again.train_ts, shuffled.train_ts)  # one permutation per seed
    assert not np.array_equal(permute_train_ts(enc, seed=2020).train_ts, shuffled.train_ts)
    # global: timestamps move between users, not just within a user
    users = enc.train[:, 0]
    moved = [u for u in np.unique(users) if sorted(shuffled.train_ts[users == u]) != sorted(original[users == u])]
    assert moved
    ckg, ckg_perm = build_ckg(enc, no_kg(enc)), build_ckg(shuffled, no_kg(enc))
    assert (ckg.t_ref, ckg.dt_den) == (ckg_perm.t_ref, ckg_perm.dt_den)
    assert np.array_equal(
        np.sort(ckg.edge_ts[~np.isnan(ckg.edge_ts)]), np.sort(ckg_perm.edge_ts[~np.isnan(ckg_perm.edge_ts)])
    )
    assert not np.array_equal(ckg.edge_ts, ckg_perm.edge_ts, equal_nan=True)


# --- scripts/run.py -------------------------------------------------------------------------------------------------


@pytest.fixture
def tsetup(synthetic, tmp_path):
    split = temporal_split(synthetic, SplitConfig(kcore=2))
    cfg = {"name": "synthetic", "run_name": RUN_NAME, "processed_dir": str(tmp_path / "processed")}
    save_split(split, cfg, {})
    _, meta = load_saved(cfg)
    enc = encode(split)
    write_kg(tmp_path / "processed", enc, meta)
    data_cfg = tmp_path / "data.yaml"
    data_cfg.write_text(yaml.safe_dump(cfg))
    models = tmp_path / "models"
    models.mkdir()
    common = {"seeds": [1, 2], "kg": {"variant": "default"}, "fixed": FIXED}
    bodies = {
        "kgat": {"name": "kgat", **common, "grid": GRID},
        "tkgat": {"name": "tkgat", **common, "from_kgat": "kgat", "grid": [{"gamma": 1}, {"gamma": 3}]},
        "tkgat_perm": {"name": "tkgat", "control": "permute_train_ts", **common, "from_tkgat": "tkgat"},
    }
    for tag, body in bodies.items():
        (models / f"{tag}.yaml").write_text(yaml.safe_dump(body))
    runs = tmp_path / "runs" / RUN_NAME
    runs.mkdir(parents=True)
    (runs / "kgat_summary.json").write_text(json.dumps({"selected": GRID[1]}))  # the point KGAT selected on val
    return SimpleNamespace(tmp=tmp_path, data=data_cfg, models=models, runs=runs, enc=enc, split=split, meta=meta)


def run_tag(ts, tag, *extra, script=None):
    (script or load_run_script()).main(
        ["--data", str(ts.data), "--model", str(ts.models / f"{tag}.yaml"), "--device", "cpu"]
        + ["--registry", str(ts.tmp / "registry.csv"), "--runs-dir", str(ts.tmp / "runs"), *extra]
    )
    return pd.read_csv(ts.tmp / "registry.csv")


def configs(reg, tag, phase):
    rows = reg[(reg["model"] == tag) & (reg["phase"] == phase)]
    return [(int(r["seed"]), json.loads(r["config"])) for _, r in rows.iterrows()]


def test_val_rows_record_gamma_and_the_time_constants(tsetup):
    reg = run_tag(tsetup, "tkgat", "--val-only")
    assert set(reg["phase"]) == {"val"} and len(reg) == 4  # 2 gammas x 2 seeds
    enc = tsetup.enc
    for _, config in configs(reg, "tkgat", "val"):
        assert {k: config[k] for k in GRID[1]} == GRID[
            1
        ]  # (lr, l2, ...) of the point KGAT selected, not written by hand
        assert config["gamma"] in (1, 3)
        assert config["t_ref"] == enc.train_ts.max() and config["dt_den"] == enc.train_ts.max() - enc.train_ts.min()
        assert "perm_seed" not in config
    assert {c["gamma"] for _, c in configs(reg, "tkgat", "val")} == {1, 3}
    summary = json.loads((tsetup.runs / "tkgat_val_summary.json").read_text())
    assert summary["tkgat"]["base_point"] == GRID[1] and summary["tkgat"]["control"] is None
    assert summary["tkgat"]["t_ref"] == enc.train_ts.max() and summary["selected"]["gamma"] in (1, 3)
    history = pd.read_csv(tsetup.runs / "tkgat_0_seed1_history.csv")
    assert history["gamma"].iloc[0] == 1 and history["t_ref"].iloc[0] == enc.train_ts.max()
    assert history["perm_seed"].isna().all()


def test_the_full_run_scores_the_selected_gamma_once_with_the_same_anchor(tsetup):
    reg = run_tag(tsetup, "tkgat")
    test_rows = configs(reg, "tkgat", "test")
    assert len(test_rows) == 2 and len({c["gamma"] for _, c in test_rows}) == 1  # one gamma, one row per seed
    selected = json.loads((tsetup.runs / "tkgat_summary.json").read_text())
    assert selected["selected"]["gamma"] == test_rows[0][1]["gamma"]
    val_anchor = {(c["t_ref"], c["dt_den"]) for _, c in configs(reg, "tkgat", "val")}
    assert {(c["t_ref"], c["dt_den"]) for _, c in test_rows} == val_anchor  # the same t_ref at val and at test
    assert selected["kg"]["t2"]["t_ref"] == selected["kg"]["t1"]["t_ref"]
    assert "test_mean_std" in selected and (tsetup.runs / "tkgat_seed1.npz").exists()


def test_control_permutes_train_ts_per_seed_and_records_the_permutation_seed(tsetup):
    run_tag(tsetup, "tkgat")
    seen = {}
    script = load_run_script()
    real = script.fit_kgat

    def spy(data, ckg, model_cfg, fit_cfg, seed):
        seen.setdefault(seed, []).append(ckg.edge_ts.copy())
        return real(data, ckg, model_cfg, fit_cfg, seed)

    script.fit_kgat = spy
    reg = run_tag(tsetup, "tkgat_perm", script=script)
    kg = script.kgat_train_graph(
        yaml.safe_load((tsetup.models / "tkgat.yaml").read_text()), yaml.safe_load(tsetup.data.read_text()), tsetup.enc,
        tsetup.meta, False,
    )[0]  # fmt: skip
    plain = build_ckg(tsetup.enc, kg)
    real_rows = {s: c for s, c in configs(reg, "tkgat", "test")}
    for seed, config in configs(reg, "tkgat_perm", "val") + configs(reg, "tkgat_perm", "test"):
        assert config["perm_seed"] == seed  # recorded, and it is the run seed
        assert (config["t_ref"], config["dt_den"]) == (real_rows[seed]["t_ref"], real_rows[seed]["dt_den"])
    for seed in (1, 2):
        expected = build_ckg(permute_train_ts(tsetup.enc, seed), kg).edge_ts
        assert len(seen[seed]) == 1  # the control has the single selected point
        assert np.array_equal(seen[seed][0], expected, equal_nan=True)  # this seed's permutation, nothing else
        assert not np.array_equal(seen[seed][0], plain.edge_ts, equal_nan=True)  # it is not the real timestamps
        assert np.array_equal(
            np.sort(seen[seed][0][~np.isnan(seen[seed][0])]), np.sort(plain.edge_ts[~np.isnan(plain.edge_ts)])
        )
    assert not np.array_equal(seen[1][0], seen[2][0], equal_nan=True)  # one permutation per seed
    # the control re-runs the point T-KGAT selected
    perm_point = {k: v for k, v in configs(reg, "tkgat_perm", "test")[0][1].items() if k != "perm_seed"}
    assert perm_point == configs(reg, "tkgat", "test")[0][1]


def test_tkgat_never_reads_test(tsetup):
    # (a) the model, the trainer and the permutation do not mention the test split at all
    for obj in (model_module.KGAT, trainer_module.fit_kgat, ckg_module.permute_train_ts, ckg_module.build_ckg):
        source = inspect.getsource(obj)
        assert "data.test" not in source and "mask_test" not in source and "test.csv" not in source, obj
    assert "test" not in inspect.signature(trainer_module.fit_kgat).parameters
    # (b) train and select with no test file and no KG at t2 on disk: the run needs neither (T-KGAT and its control)
    for name in ("test.csv", "seen_before_test.csv"):
        (tsetup.tmp / "processed" / name).unlink()
    (tsetup.tmp / "processed" / "kg" / "default" / "kg_t2.tsv").unlink()
    reg = run_tag(tsetup, "tkgat", "--val-only")
    reg = run_tag(tsetup, "tkgat_perm", "--val-only")
    assert set(reg["phase"]) == {"val"} and not list(tsetup.runs.glob("*.npz"))
    assert not (tsetup.runs / "tkgat_summary.json").exists()


def test_a_run_trained_before_is_resumed_without_training_or_new_rows(tsetup, monkeypatch):
    run_tag(tsetup, "tkgat")
    before = pd.read_csv(tsetup.tmp / "registry.csv")
    script = load_run_script()

    def no_training(*args, **kwargs):
        raise AssertionError("a finished run must not be trained again")

    monkeypatch.setattr(script, "fit_kgat", no_training)
    run_tag(tsetup, "tkgat", "--resume", script=script)
    pd.testing.assert_frame_equal(before, pd.read_csv(tsetup.tmp / "registry.csv"))


def edit(ts, tag, **changes):
    path = ts.models / f"{tag}.yaml"
    body = yaml.safe_load(path.read_text())
    for key, value in changes.items():
        if value is None:
            body.pop(key, None)
        else:
            body[key] = value
    path.write_text(yaml.safe_dump(body))


def test_t_kgat_needs_a_finished_kgat_run(tsetup):
    (tsetup.runs / "kgat_summary.json").unlink()
    with pytest.raises(FileNotFoundError, match="run kgat first"):
        run_tag(tsetup, "tkgat", "--val-only")


def test_a_val_only_run_may_use_the_val_summary_of_kgat(tsetup):
    (tsetup.runs / "kgat_summary.json").rename(tsetup.runs / "kgat_val_summary.json")
    run_tag(tsetup, "tkgat", "--val-only")


def test_an_official_run_refuses_the_val_summary_of_kgat(tsetup):
    (tsetup.runs / "kgat_summary.json").rename(tsetup.runs / "kgat_val_summary.json")
    with pytest.raises(FileNotFoundError, match="official run"):
        run_tag(tsetup, "tkgat")
    assert not (tsetup.tmp / "registry.csv").exists()  # it stopped before training anything


def test_an_official_run_without_any_kgat_summary_says_to_run_kgat(tsetup):
    (tsetup.runs / "kgat_summary.json").unlink()
    with pytest.raises(FileNotFoundError, match="run kgat first") as err:
        run_tag(tsetup, "tkgat")
    assert "official run" not in str(err.value)  # nothing to say about a val summary that does not exist


def test_an_official_control_refuses_the_val_summary_of_tkgat(tsetup):
    run_tag(tsetup, "tkgat", "--val-only")  # leaves tkgat_val_summary.json only
    assert not (tsetup.runs / "tkgat_summary.json").exists()
    with pytest.raises(FileNotFoundError, match="official run"):
        run_tag(tsetup, "tkgat_perm")
    run_tag(tsetup, "tkgat_perm", "--val-only")  # the val-only control may use it


@pytest.mark.parametrize(
    "tag, changes, message",
    [
        ("tkgat", {"fixed": {**FIXED, "layers": [4, 4]}}, "fixed differs"),
        ("tkgat", {"seeds": [1, 2, 3]}, "seeds differs"),
        ("tkgat", {"kg": {"variant": "static"}}, "kg differs"),
        ("tkgat", {"grid": [{"gamma": 0}]}, "gamma > 0"),
        ("tkgat", {"grid": [{"gamma": 1, "lr": 0.5}]}, "gamma > 0"),
        ("tkgat", {"from_kgat": None}, "needs from_kgat"),
        ("tkgat", {"from_kgat": "tkgat_perm"}, "not a kgat config"),
        ("tkgat", {"control": "shuffle_everything"}, "control must be one of"),
        ("tkgat_perm", {"from_tkgat": None}, "needs from_tkgat"),
        ("tkgat_perm", {"grid": [{"gamma": 1}]}, "remove grid"),
        ("tkgat_perm", {"from_tkgat": "kgat"}, "not a tkgat config"),
    ],
)
def test_the_configs_are_checked_against_the_config_they_extend(tsetup, tag, changes, message):
    run_tag(tsetup, "tkgat", "--val-only")  # the point the control would need
    edit(tsetup, tag, **changes)
    with pytest.raises(ValueError, match=message):
        run_tag(tsetup, tag, "--val-only")


@pytest.mark.parametrize("tag", ["tkgat", "tkgat_perm"])
def test_the_shipped_configs_extend_kgat_yaml(tag):
    cfg = yaml.safe_load((ROOT / "configs" / "model" / f"{tag}.yaml").read_text())
    ref = yaml.safe_load((ROOT / "configs" / "model" / "kgat.yaml").read_text())
    assert cfg["name"] == "tkgat" and all(cfg[k] == ref[k] for k in ("fixed", "kg", "seeds"))
    assert cfg["seeds"] == [2019, 2020, 2021, 2022, 2023]
    if tag == "tkgat":
        assert cfg["from_kgat"] == "kgat" and [p["gamma"] for p in cfg["grid"]] == [1, 3, 10]
        assert "control" not in cfg
        for point in cfg["grid"]:  # lr and l2 are never written here: they come from the KGAT run
            split_params({**ref["grid"][0], **point, **cfg["fixed"]}, "cpu")
    else:
        assert cfg["control"] == "permute_train_ts" and cfg["from_tkgat"] == "tkgat" and "grid" not in cfg
