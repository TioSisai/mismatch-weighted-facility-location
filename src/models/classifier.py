"""Frame-level multilabel MLP (GELU, no dropout) and per-round training components."""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import torch
from skorch import NeuralNet
from skorch.callbacks import Callback, LRScheduler
from timm.layers import Mlp
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from .metrics import frame_wise_map

# The class name is written to the optimizer field in config.json.
OPTIMIZER_CLS = torch.optim.Adam

# Metric for selecting the best weights, written to config.json.
MODEL_SELECTION_METRIC = "val_mAP"


def make_warmup_cosine(warmup_steps: int, total_steps: int):
    """Construct a LambdaLR multiplier function with stepwise linear warmup and cosine annealing.

    Args:
        warmup_steps: Number of warmup steps, reaching 1 at step ``warmup_steps-1``.
        total_steps: Total number of training steps.

    Returns:
        ``lr_lambda(step) -> float``.
    """

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denom = max(1, total_steps - warmup_steps)
        progress = min(max(float(step - warmup_steps) / float(denom), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


class FrameMLPClassifier(NeuralNet):
    """Frame-level classifier using skorch to wrap timm Mlp.

    Args:
        num_classes: Number of output classes.
        in_features: Frame embedding dimension, defaulting to 2048.
        hidden_features: Hidden dimension, defaulting to 512.
        **kwargs: Passed to skorch.NeuralNet, with BCEWithLogitsLoss as the default loss.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        in_features: int = 2048,
        hidden_features: int = 512,
        **kwargs,
    ) -> None:
        kwargs.setdefault("criterion", nn.BCEWithLogitsLoss)
        super().__init__(
            module=Mlp,
            module__in_features=in_features,
            module__hidden_features=hidden_features,
            module__out_features=num_classes,
            **kwargs,
        )

    @torch.inference_mode()
    def predict_frame_proba(self, X) -> np.ndarray:
        """Return per-frame sigmoid probabilities in batches according to iterator_valid.

        Each call consumes one global torch random draw and preserves three-dimensional multilabel output.

        Args:
            X: Frame embeddings, ``[N, F, in_features]``.

        Returns:
            float32 probabilities, ``[N, F, num_classes]``.
        """
        self.check_is_fitted()
        self.module_.eval()
        logits = self.forward(X)
        return torch.sigmoid(logits).cpu().numpy()


class BestValCheckpoint(Callback):
    """Save the best weights by validation mAP each epoch, retaining the earlier epoch in a tie.

    NaN does not trigger an update, and the last epoch is saved if no valid value occurs throughout training.

    Args:
        x_val: Validation frame embeddings, ``[M, F, in_features]``.
        y_val: Validation frame multilabel targets, ``[M, F, C]``.
        val_valid_lengths: Valid frame counts, ``[M]``.

    Attributes:
        best_map: Best validation mAP, or NaN if no valid value is available.
        best_per_class: Corresponding per-class AP, ``[C]``.
        best_epoch: Corresponding epoch, starting at 1.
        best_state: Deep copy of the best state_dict, on the same device as the model.
    """

    def __init__(self, x_val, y_val, val_valid_lengths) -> None:
        self.x_val = x_val
        self.y_val = y_val
        self.val_valid_lengths = val_valid_lengths

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        self.best_map = float("nan")
        self.best_per_class = None
        self.best_epoch = None
        self.best_state = None
        self._last_per_class = None
        self._last_map = float("nan")

    def on_epoch_end(self, net, **kwargs):
        # Preserve the skorch inference path to maintain the order of global torch random number consumption.
        per_class, macro = frame_wise_map(
            net.predict_frame_proba(self.x_val), self.y_val, self.val_valid_lengths
        )
        self._last_per_class, self._last_map = per_class, macro
        if not math.isnan(macro) and (math.isnan(self.best_map) or macro > self.best_map):
            self._record(net, per_class, macro)

    def on_train_end(self, net, X=None, y=None, **kwargs):
        if self.best_state is None:
            self._record(net, self._last_per_class, self._last_map)

    def _record(self, net, per_class, macro) -> None:
        self.best_map = macro
        self.best_per_class = per_class
        self.best_epoch = len(net.history)
        self.best_state = deepcopy(net.module_.state_dict())


def build_training_net(
    *,
    num_classes: int,
    hidden_features: int,
    num_labeled: int,
    x_val,
    y_val,
    val_valid_lengths,
    lr: float,
    warmup_ratio: float,
    num_epochs: int,
    batch_size: int,
    infer_batch_size: int,
    device,
) -> tuple[FrameMLPClassifier, BestValCheckpoint]:
    """Build the classifier, warmup and cosine scheduler, and best validation weight callback for one round.

    Args:
        num_classes: Number of classes.
        hidden_features: Hidden dimension.
        num_labeled: Number of labeled segments in this round, used to calculate training steps.
        x_val: Validation frame embeddings, ``[M, F, in_features]``.
        y_val: Validation frame multilabel targets, ``[M, F, C]``.
        val_valid_lengths: Valid frame counts, ``[M]``.
        lr: Peak learning rate.
        warmup_ratio: Fraction of warmup steps.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        infer_batch_size: Inference batch size for validation, testing, and querying.
        device: torch device.

    Returns:
        ``(net, val_ckpt)``, the untrained classifier and its best validation weight callback.
    """
    steps_per_epoch = math.ceil(num_labeled / batch_size)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(warmup_ratio * total_steps)
    lr_sched = LRScheduler(
        policy=LambdaLR,
        lr_lambda=make_warmup_cosine(warmup_steps, total_steps),
        step_every="batch",
    )
    val_ckpt = BestValCheckpoint(x_val, y_val, val_valid_lengths)
    # Preserve the DataLoader setting from the paper's experiments that does not shuffle samples.
    net = FrameMLPClassifier(
        num_classes,
        hidden_features=hidden_features,
        max_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        iterator_valid__batch_size=infer_batch_size,
        optimizer=OPTIMIZER_CLS,
        train_split=None,
        callbacks=[("lr_sched", lr_sched), ("val_ckpt", val_ckpt)],
        device=device,
        verbose=0,
    )
    return net, val_ckpt


def build_inference_net(
    *,
    num_classes: int,
    hidden_features: int,
    state_dict: dict,
    batch_size: int,
    infer_batch_size: int,
    device,
) -> FrameMLPClassifier:
    """Create a classifier and load the previous round's best weights for query inference.

    The random numbers consumed by initialize() belong to the experiment's random stream, and directly reusing the training network would change subsequent results.

    Args:
        num_classes: Number of classes.
        hidden_features: Hidden dimension.
        state_dict: best_state from the previous round.
        batch_size: Training batch size, used only to construct the network.
        infer_batch_size: Inference batch size.
        device: torch device.

    Returns:
        FrameMLPClassifier with weights loaded.
    """
    net = FrameMLPClassifier(
        num_classes,
        hidden_features=hidden_features,
        batch_size=batch_size,
        iterator_valid__batch_size=infer_batch_size,
        device=device,
    )
    net.initialize()
    net.module_.load_state_dict(state_dict)
    return net
