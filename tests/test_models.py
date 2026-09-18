"""Test the classification head, training components, and evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.models.classifier import (
    BestValCheckpoint,
    FrameMLPClassifier,
    build_inference_net,
    build_training_net,
    make_warmup_cosine,
)
from src.models.metrics import cumulative_aulc, frame_wise_map



def test_frame_wise_map_perfect_ranking():
    proba = np.array([[[0.9, 0.1], [0.8, 0.2]], [[0.2, 0.95], [0.1, 0.85]]], dtype=np.float32)
    label = np.array([[[1, 0], [1, 0]], [[0, 1], [0, 1]]], dtype=np.float32)
    per_class, macro = frame_wise_map(proba, label, np.array([2, 2]))
    assert per_class.dtype == np.float64
    assert np.allclose(per_class, [1.0, 1.0], atol=1e-5) and abs(macro - 1.0) < 1e-5


def test_frame_wise_map_excludes_padding_and_absent_class():
    proba = np.array([[[0.9, 0.5], [0.99, 0.99]]], dtype=np.float32)
    label = np.array([[[1, 0], [1, 0]]], dtype=np.float32)
    per_class, macro = frame_wise_map(proba, label, np.array([1]))
    assert np.isnan(per_class[1]) and abs(macro - per_class[0]) < 1e-6


def test_frame_wise_map_all_classes_absent_is_nan():
    proba = np.full((2, 3, 2), 0.5, dtype=np.float32)
    label = np.zeros((2, 3, 2), dtype=np.float32)
    per_class, macro = frame_wise_map(proba, label, np.array([3, 2]))
    assert np.isnan(per_class).all() and math.isnan(macro)


def test_cumulative_aulc_normalized():
    assert math.isnan(cumulative_aulc([100], [0.5]))
    assert math.isnan(cumulative_aulc([100, 100], [0.4, 0.6]))
    assert abs(cumulative_aulc([100, 200], [0.4, 0.6]) - 0.5) < 1e-9
    assert abs(cumulative_aulc([0, 100, 200], [0.2, 0.4, 0.6]) - 0.4) < 1e-9




def _toy_frames(n, num_classes, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 31, 2048)).astype(np.float32)
    y = (rng.random((n, 31, num_classes)) > 0.6).astype(np.float32)
    return x, y


def test_frame_mlp_train_and_proba_shape():
    x, y = _toy_frames(16, 5)
    net = FrameMLPClassifier(num_classes=5, max_epochs=1, lr=1e-3, batch_size=8,
                             train_split=None, device="cpu")
    net.fit(x, y)
    proba = net.predict_frame_proba(x[:4])
    assert proba.shape == (4, 31, 5) and proba.dtype == np.float32
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_frame_mlp_default_dims_and_no_shuffle():
    net = FrameMLPClassifier(num_classes=10, device="cpu")
    net.initialize()
    assert net.module_.fc1.in_features == 2048
    assert net.module_.fc1.out_features == 512
    assert net.module_.fc2.out_features == 10
    assert not net.get_params_for("iterator_train").get("shuffle", False)


def test_make_warmup_cosine_schedule():
    lr_lambda = make_warmup_cosine(warmup_steps=4, total_steps=20)
    assert [lr_lambda(s) for s in range(4)] == [0.25, 0.5, 0.75, 1.0]
    assert lr_lambda(4) == 1.0
    assert abs(lr_lambda(12) - 0.5) < 1e-12
    assert abs(lr_lambda(20)) < 1e-12
    assert make_warmup_cosine(0, 10)(0) == 1.0


def test_build_training_net_snapshots_best_state():
    x, y = _toy_frames(8, 3)
    x_val, y_val = _toy_frames(4, 3, seed=1)
    net, val_ckpt = build_training_net(
        num_classes=3, hidden_features=32, num_labeled=8, x_val=x_val, y_val=y_val,
        val_valid_lengths=np.full(4, 31), lr=1e-3, warmup_ratio=0.2, num_epochs=3,
        batch_size=4, infer_batch_size=3, device="cpu",
    )
    assert [name for name, _ in net.callbacks] == ["lr_sched", "val_ckpt"]
    assert net.get_params_for("iterator_valid")["batch_size"] == 3
    net.fit(x, y)
    assert 1 <= val_ckpt.best_epoch <= 3
    assert np.isfinite(val_ckpt.best_map) and val_ckpt.best_per_class.shape == (3,)
    live = net.module_.state_dict()
    assert all(val_ckpt.best_state[k].data_ptr() != live[k].data_ptr() for k in live)

    net.module_.load_state_dict(val_ckpt.best_state)
    _, macro = frame_wise_map(net.predict_frame_proba(x_val), y_val, np.full(4, 31))
    assert macro == val_ckpt.best_map


def test_best_val_checkpoint_falls_back_to_last_epoch_when_val_map_is_nan():
    x, y = _toy_frames(8, 2)
    x_val, _ = _toy_frames(3, 2, seed=1)
    y_val = np.zeros((3, 31, 2), dtype=np.float32)
    net, val_ckpt = build_training_net(
        num_classes=2, hidden_features=16, num_labeled=8, x_val=x_val, y_val=y_val,
        val_valid_lengths=np.full(3, 31), lr=1e-3, warmup_ratio=0.2, num_epochs=2,
        batch_size=4, infer_batch_size=8, device="cpu",
    )
    net.fit(x, y)
    assert math.isnan(val_ckpt.best_map) and val_ckpt.best_epoch == 2
    for key, value in net.module_.state_dict().items():
        assert torch.equal(val_ckpt.best_state[key], value)


def test_best_val_checkpoint_strictly_greater(monkeypatch):
    scores = iter([float("nan"), 0.5, 0.5, 0.7, 0.7])
    monkeypatch.setattr(
        "src.models.classifier.frame_wise_map",
        lambda *args: (np.zeros(2), next(scores)),
    )

    class FakeNet:
        def __init__(self):
            self.history = []
            self.module_ = torch.nn.Linear(2, 2)

        def predict_frame_proba(self, x):
            return x

    net = FakeNet()
    ckpt = BestValCheckpoint(None, None, None)
    ckpt.on_train_begin(net)
    for _ in range(5):
        net.history.append({})
        ckpt.on_epoch_end(net)
    ckpt.on_train_end(net)
    assert ckpt.best_map == 0.7 and ckpt.best_epoch == 4


def test_build_inference_net_loads_state_and_consumes_init_rng():
    reference = FrameMLPClassifier(num_classes=3, hidden_features=16, device="cpu")
    reference.initialize()
    state = {k: v.clone() for k, v in reference.module_.state_dict().items()}

    torch.manual_seed(123)
    net = build_inference_net(num_classes=3, hidden_features=16, state_dict=state,
                              batch_size=4, infer_batch_size=5, device="cpu")
    after_build = torch.rand(1)
    for key, value in state.items():
        assert torch.equal(net.module_.state_dict()[key], value)
    assert net.get_params_for("iterator_valid")["batch_size"] == 5

    torch.manual_seed(123)
    FrameMLPClassifier(num_classes=3, hidden_features=16, device="cpu").initialize()
    assert torch.equal(after_build, torch.rand(1))


@pytest.mark.parametrize("infer_batch_size", [1, 7])
def test_predict_frame_proba_consumes_one_rng_draw_per_call(infer_batch_size):
    net = FrameMLPClassifier(num_classes=2, hidden_features=8, device="cpu",
                             iterator_valid__batch_size=infer_batch_size)
    net.initialize()
    x, _ = _toy_frames(5, 2)

    torch.manual_seed(0)
    net.predict_frame_proba(x)
    after_predict = torch.rand(1)

    torch.manual_seed(0)
    torch.empty((), dtype=torch.int64).random_()
    assert torch.equal(after_predict, torch.rand(1))
