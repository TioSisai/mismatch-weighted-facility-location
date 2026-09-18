"""Test experiment configuration, training round by round, and resuming with synthetic caches."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest
import torch

import src.learner as learner_module
from src.data.arrays import load_frame_data
from src.learner import (
    OVERALL_FIELDNAMES,
    ActiveLearner,
    check_or_write_config,
    config_dir,
    config_hash,
    results_csv_complete,
    strategy_config,
)
from src.models.classifier import FrameMLPClassifier

_CONFIG_KWARGS = dict(
    dataset="DESED", strategy="mw-fl", step_size=25, max_iter=20, hidden_features=512,
    lr=1e-3, warmup_ratio=0.2, num_epochs=25, batch_size=16, infer_batch_size=128,
    num_classes=10, n_train=10000, device="cuda",
)


@pytest.fixture
def frame_data(synthetic_cache):
    """Synthetic FrameData with 40 training segments and 3 classes."""
    return load_frame_data(synthetic_cache)


def _learner(frame_data, output_root, **overrides):
    kwargs = dict(
        data=frame_data, dataset="Toy", strategy_name="mw-fl", output_root=output_root,
        step_size=4, max_iter=3, hidden_features=32, lr=1e-3, warmup_ratio=0.2,
        num_epochs=2, seed=0, device="cpu", batch_size=4, infer_batch_size=8,
    )
    kwargs.update(overrides)
    return ActiveLearner(**kwargs)


def _read_rows(seed_dir):
    with (seed_dir / "results.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)




def test_config_records_implementation_fields():
    config = strategy_config(**_CONFIG_KWARGS)
    assert config["optimizer"] == "adam"
    assert config["model_selection"] == "val_mAP"
    assert isinstance(config["backend"], str) and config["backend"]


def test_config_hash_is_order_insensitive():
    config = strategy_config(**_CONFIG_KWARGS)
    shuffled = dict(reversed(list(config.items())))
    assert list(shuffled) != list(config)
    assert config_hash(shuffled) == config_hash(config)
    assert len(config_hash(config)) == 12


@pytest.mark.parametrize("field,value", [
    ("dataset", "DataSED"), ("strategy", "mw-ft"), ("step_size", 50), ("max_iter", 21),
    ("hidden_features", 256), ("lr", 2e-3), ("warmup_ratio", 0.1), ("num_epochs", 50),
    ("batch_size", 32), ("infer_batch_size", 256), ("num_classes", 11),
    ("n_train", 10001), ("device", "cpu"),
])
def test_config_hash_is_sensitive_to_every_field(field, value):
    base = strategy_config(**_CONFIG_KWARGS)
    changed = strategy_config(**{**_CONFIG_KWARGS, field: value})
    assert config_hash(changed) != config_hash(base)


def test_config_dir_layout(tmp_path):
    config = strategy_config(**_CONFIG_KWARGS)
    assert config_dir(tmp_path, config) == tmp_path / "DESED" / "mw-fl" / config_hash(config)


def test_check_or_write_config_fails_fast_on_mismatch(tmp_path):
    config = strategy_config(**_CONFIG_KWARGS)
    check_or_write_config(tmp_path / "cfg", config)
    check_or_write_config(tmp_path / "cfg", config)
    assert json.loads((tmp_path / "cfg" / "config.json").read_text()) == config
    with pytest.raises(ValueError, match="does not match"):
        check_or_write_config(tmp_path / "cfg", {**config, "lr": 1.0})


def test_results_csv_complete_checks_schema_and_rows(tmp_path):
    path = tmp_path / "results.csv"
    assert not results_csv_complete(path, 2)
    header = ",".join(OVERALL_FIELDNAMES + ["queried_idxes_in_latest_iteration"])
    path.write_text(header + "\n0,4,,,,,[]\n1,8,,,,,[]\n")
    assert results_csv_complete(path, 2)
    assert not results_csv_complete(path, 3)
    path.write_text("iteration,num_labeled_samples,val_mAP\n0,4,\n1,8,\n")
    assert not results_csv_complete(path, 2)




def test_learner_rejects_insufficient_pool(frame_data, tmp_path):
    with pytest.raises(ValueError, match="not enough samples"):
        _learner(frame_data, tmp_path, step_size=21, max_iter=2)


def test_learner_writes_results_and_checkpoints(frame_data, tmp_path):
    learner = _learner(frame_data, tmp_path / "out")
    seed_dir = learner.run()
    assert seed_dir == learner.config_dir / "seed_0"
    assert seed_dir.parent.parent == tmp_path / "out" / "Toy" / "mw-fl"
    assert json.loads((seed_dir.parent / "config.json").read_text()) == learner.config

    fieldnames, rows = _read_rows(seed_dir)
    assert fieldnames == (
        OVERALL_FIELDNAMES
        + [f"val_mAP_class_{c}" for c in range(3)]
        + [f"test_mAP_class_{c}" for c in range(3)]
        + ["queried_idxes_in_latest_iteration"]
    )
    assert [row["iteration"] for row in rows] == ["0", "1", "2"]
    assert [row["num_labeled_samples"] for row in rows] == ["4", "8", "12"]
    assert rows[0]["val_AULC"] == "" and rows[0]["test_AULC"] == ""  # AULC is undefined for a single point

    labeled: set[int] = set()
    for iteration, row in enumerate(rows):
        selected = json.loads(row["queried_idxes_in_latest_iteration"])
        assert len(selected) == 4 and len(set(selected)) == 4
        assert labeled.isdisjoint(selected)
        labeled |= set(selected)
        for key in ("val_mAP", "test_mAP") + (("val_AULC", "test_AULC") if iteration else ()):
            assert len(row[key].split(".")[1]) == 6 and 0.0 <= float(row[key]) <= 1.0

        ckpt = seed_dir / (
            f"iter={iteration}-labeled={len(labeled)}-val_mAP={float(row['val_mAP']):.6f}.pth"
        )
        state = torch.load(ckpt)
        assert state["fc2.weight"].shape == (3, 32)
        assert all(t.device.type == "cpu" for t in state.values())


def test_learner_resume_skips_complete_and_reruns_incomplete(frame_data, tmp_path):
    seed_dir = _learner(frame_data, tmp_path).run()
    results_csv = seed_dir / "results.csv"
    mtime = results_csv.stat().st_mtime_ns
    assert _learner(frame_data, tmp_path).run() == seed_dir
    assert results_csv.stat().st_mtime_ns == mtime

    full_text = results_csv.read_text()
    results_csv.write_text("".join(full_text.splitlines(keepends=True)[:2]))
    (seed_dir / "stale.txt").write_text("x")
    _learner(frame_data, tmp_path).run()
    assert results_csv.read_text() == full_text
    assert not (seed_dir / "stale.txt").exists()


def test_learner_same_seed_reproducible(frame_data, tmp_path):
    seed_a = _learner(frame_data, tmp_path / "a", strategy_name="mf-ft").run()
    seed_b = _learner(frame_data, tmp_path / "b", strategy_name="mf-ft").run()
    assert (seed_a / "results.csv").read_text() == (seed_b / "results.csv").read_text()
    ckpts_a = sorted(p.name for p in seed_a.glob("*.pth"))
    assert ckpts_a == sorted(p.name for p in seed_b.glob("*.pth")) and len(ckpts_a) == 3
    for name in ckpts_a:
        state_a, state_b = torch.load(seed_a / name), torch.load(seed_b / name)
        assert all(torch.equal(state_a[k], state_b[k]) for k in state_a)

    seed_1 = _learner(frame_data, tmp_path / "a", strategy_name="mf-ft", seed=1).run()
    assert seed_1.parent == seed_a.parent and seed_1.name == "seed_1"


def test_learner_passes_expected_query_inputs(frame_data, tmp_path, monkeypatch):
    calls = []
    original_build = learner_module.build_query_strategy

    def spy_build(name, *, random_state):
        strategy = original_build(name, random_state=random_state)
        original_query = strategy.query

        def query(embeddings, candidates, labeled, **kwargs):
            calls.append((np.asarray(candidates), np.asarray(labeled), kwargs))
            return original_query(embeddings, candidates, labeled, **kwargs)

        strategy.query = query
        return strategy

    monkeypatch.setattr(learner_module, "build_query_strategy", spy_build)
    _learner(frame_data, tmp_path, strategy_name="mf-fl").run()

    assert len(calls) == 3
    candidates, labeled, kwargs = calls[0]
    assert kwargs["seg_proba"] is None and labeled.size == 0 and candidates.size == 40
    assert kwargs["batch_size"] == 4 and kwargs["dist_matrix"] is frame_data.pairwise_distances
    for iteration, (candidates, labeled, kwargs) in enumerate(calls[1:], start=1):
        assert labeled.size == 4 * iteration and candidates.size == 40 - 4 * iteration
        assert np.intersect1d(candidates, labeled).size == 0
        assert kwargs["seg_proba"].shape == (40, 3)
        assert kwargs["seg_labels"] is frame_data.seg_labels


def test_learner_uses_infer_batch_size_for_all_inference(frame_data, tmp_path, monkeypatch):
    seen = []
    original_predict = FrameMLPClassifier.predict_frame_proba

    def spy_predict(self, X):
        seen.append((self.batch_size, self.get_params_for("iterator_valid").get("batch_size")))
        return original_predict(self, X)

    monkeypatch.setattr(FrameMLPClassifier, "predict_frame_proba", spy_predict)
    _learner(frame_data, tmp_path, max_iter=2, batch_size=2, infer_batch_size=5).run()
    # Each round has num_epochs validation passes and one test pass, with one additional query inference pass in round 1.
    assert len(seen) == 2 * (2 + 1) + 1
    assert set(seen) == {(2, 5)}


def test_learner_rejects_invalid_selection(frame_data, tmp_path, monkeypatch):
    class DuplicateStrategy:
        def query(self, embeddings, candidates, labeled, **kwargs):
            return np.array([candidates[0]] * kwargs["batch_size"])

    monkeypatch.setattr(
        learner_module, "build_query_strategy", lambda name, *, random_state: DuplicateStrategy()
    )
    with pytest.raises(RuntimeError, match="invalid selection"):
        _learner(frame_data, tmp_path).run()


def test_learner_nan_val_map_is_blank_in_csv(frame_data, tmp_path):
    from dataclasses import replace

    no_positive = replace(frame_data, val_label=np.zeros_like(frame_data.val_label))
    seed_dir = _learner(no_positive, tmp_path, max_iter=2).run()
    _, rows = _read_rows(seed_dir)
    assert all(row["val_mAP"] == "" and row["val_AULC"] == "" for row in rows)
    assert all(not math.isnan(float(row["test_mAP"])) for row in rows)
    assert (seed_dir / "iter=0-labeled=4-val_mAP=nan.pth").is_file()
