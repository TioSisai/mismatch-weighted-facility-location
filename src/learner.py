"""Active learning training and result saving for a single dataset, strategy, and seed."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import threading
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .data.arrays import FrameData
from .data.frames import segment_max_pool
from .models.classifier import (
    MODEL_SELECTION_METRIC,
    OPTIMIZER_CLS,
    build_inference_net,
    build_training_net,
)
from .models.metrics import cumulative_aulc, frame_wise_map
from .strategies import build_query_strategy
from .strategies.backends import backend_signature

CONFIG_HASH_LEN = 12

# Overall metric columns are also used to validate the CSV header when resuming.
OVERALL_FIELDNAMES = [
    "iteration", "num_labeled_samples", "val_mAP", "val_AULC", "test_mAP", "test_AULC",
]


def strategy_config(
    *, dataset, strategy, step_size, max_iter, hidden_features, lr, warmup_ratio,
    num_epochs, batch_size, infer_batch_size, num_classes, n_train, device,
) -> dict:
    """Generate a configuration shared across seeds, written to config.json and determining the output directory hash.

    Args:
        dataset: Dataset name.
        strategy: Strategy name.
        step_size: Number of samples selected per round.
        max_iter: Total number of rounds.
        hidden_features: Hidden dimension of the classification head.
        lr: Peak learning rate.
        warmup_ratio: Fraction of warmup steps.
        num_epochs: Number of training epochs per round.
        batch_size: Training batch size.
        infer_batch_size: Inference batch size.
        num_classes: Number of classes.
        n_train: Number of segments in the training pool.
        device: torch device.

    Returns:
        JSON-serializable dictionary containing all result-related settings.
    """
    return {
        "dataset": dataset,
        "strategy": strategy,
        "step_size": int(step_size),
        "max_iter": int(max_iter),
        "hidden_features": int(hidden_features),
        "optimizer": OPTIMIZER_CLS.__name__.lower(),
        "model_selection": MODEL_SELECTION_METRIC,
        "lr": float(lr),
        "warmup_ratio": float(warmup_ratio),
        "num_epochs": int(num_epochs),
        "batch_size": int(batch_size),
        "infer_batch_size": int(infer_batch_size),
        "num_classes": int(num_classes),
        "n_train": int(n_train),
        "device": str(device),
        "backend": backend_signature(),
    }


def config_hash(config: dict) -> str:
    """Return the first CONFIG_HASH_LEN characters of the configuration's SHA-256, independent of field order."""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:CONFIG_HASH_LEN]


def config_dir(output_root, config: dict) -> Path:
    """Return ``{output_root}/{dataset}/{strategy}/{config_hash}`` without creating the directory."""
    return Path(output_root) / config["dataset"] / config["strategy"] / config_hash(config)


def check_or_write_config(cfg_dir: Path, config: dict) -> None:
    """Validate or atomically write config.json, allowing multiple processes to share the output directory.

    Raises:
        ValueError: The existing configuration differs from the supplied configuration.
    """
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "config.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                f"config.json does not match the current parameters (config hash collision): {path}\n"
                f"existing: {existing}\ncurrent: {config}"
            )
    else:
        tmp_path = cfg_dir / f".config.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)


def results_csv_complete(results_csv: Path, max_iter: int) -> bool:
    """Check that the header starts with OVERALL_FIELDNAMES and the results contain exactly max_iter rows."""
    if not results_csv.is_file():
        return False
    with results_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if (reader.fieldnames or [])[:len(OVERALL_FIELDNAMES)] != OVERALL_FIELDNAMES:
            return False
        return len(list(reader)) == max_iter


class ActiveLearner:
    """Select samples and train from scratch each round, evaluating the test set with the best weights by validation mAP.

    Args:
        data: FrameData reusable across strategies and seeds.
        dataset: Dataset name.
        strategy_name: Query strategy name.
        output_root: Output root directory.
        step_size: Number of samples selected per round, including the cold start.
        max_iter: Total number of rounds.
        hidden_features: Hidden dimension of the classification head.
        lr: Peak learning rate.
        warmup_ratio: Fraction of warmup steps.
        num_epochs: Number of training epochs per round.
        seed: Seed for global random numbers and the query strategy.
        device: torch device.
        batch_size: Training batch size.
        infer_batch_size: Inference batch size for validation, testing, and querying.
        pbar_position: Progress bar line number, with 0 reserved for the overall progress bar.

    Raises:
        ValueError: The total selection budget exceeds the training pool size.
    """

    def __init__(
        self,
        *,
        data: FrameData,
        dataset: str,
        strategy_name: str,
        output_root,
        step_size: int,
        max_iter: int,
        hidden_features: int,
        lr: float,
        warmup_ratio: float,
        num_epochs: int,
        seed: int,
        device,
        batch_size: int,
        infer_batch_size: int,
        pbar_position: int = 1,
    ) -> None:
        self.data = data
        self.dataset = dataset
        self.strategy_name = strategy_name
        self.step_size = int(step_size)
        self.max_iter = int(max_iter)
        self.hidden_features = int(hidden_features)
        self.lr = float(lr)
        self.warmup_ratio = float(warmup_ratio)
        self.num_epochs = int(num_epochs)
        self.seed = int(seed)
        self.device = device
        self.batch_size = int(batch_size)
        self.infer_batch_size = int(infer_batch_size)
        self.pbar_position = pbar_position

        required = self.max_iter * self.step_size
        if required > data.n_train:
            raise ValueError(
                f"not enough samples: max_iter*step_size={required} required, train pool only has {data.n_train}"
            )

        self.config = strategy_config(
            dataset=self.dataset,
            strategy=self.strategy_name,
            step_size=self.step_size,
            max_iter=self.max_iter,
            hidden_features=self.hidden_features,
            lr=self.lr,
            warmup_ratio=self.warmup_ratio,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            infer_batch_size=self.infer_batch_size,
            num_classes=data.num_classes,
            n_train=data.n_train,
            device=self.device,
        )
        self.config_dir = config_dir(output_root, self.config)
        self.seed_dir = self.config_dir / f"seed_{self.seed}"
        self.results_csv = self.seed_dir / "results.csv"

    def _seed_everything(self) -> None:
        """Initialize numpy and torch (including CUDA) random numbers, called only once per experiment."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _select_samples(self, labeled: list[int], prev_state: dict | None) -> list[int]:
        """Select step_size segments from the unlabeled pool, using a cold start when prev_state is empty."""
        labeled_arr = np.asarray(labeled, dtype=int)
        unlabeled = np.setdiff1d(np.arange(self.data.n_train), labeled_arr)
        seg_proba = None
        if prev_state is not None:
            net = build_inference_net(
                num_classes=self.data.num_classes,
                hidden_features=self.hidden_features,
                state_dict=prev_state,
                batch_size=self.batch_size,
                infer_batch_size=self.infer_batch_size,
                device=self.device,
            )
            frame_proba = net.predict_frame_proba(self.data.train_embedding)  # [N, F, C]
            seg_proba = segment_max_pool(frame_proba, self.data.train_valid_lengths)  # [N, C]
        # Rebuild the strategy with the same seed each round so its random stream starts at the same position.
        strategy = build_query_strategy(self.strategy_name, random_state=self.seed)
        selected = strategy.query(
            self.data.seg_embedding,
            unlabeled,
            labeled_arr,
            seg_proba=seg_proba,
            seg_labels=self.data.seg_labels,
            batch_size=self.step_size,
            dist_matrix=self.data.pairwise_distances,
        )
        return [int(i) for i in np.asarray(selected)]

    def _assert_selection(self, selected: list[int], labeled: list[int]) -> None:
        """Check the selection count, uniqueness within the batch, and disjointness from the labeled set."""
        unique = set(selected)
        if (
            len(selected) != self.step_size
            or len(unique) != len(selected)
            or not unique.isdisjoint(labeled)
        ):
            raise RuntimeError(
                f"strategy {self.strategy_name} returned an invalid selection (expected {self.step_size} distinct"
                f" unlabeled indices): {selected}"
            )

    def _csv_fieldnames(self) -> list[str]:
        return (
            OVERALL_FIELDNAMES
            + [f"val_mAP_{c}" for c in self.data.class_names]
            + [f"test_mAP_{c}" for c in self.data.class_names]
            + ["queried_idxes_in_latest_iteration"]
        )

    @staticmethod
    def _fmt(value) -> str:
        """Format metrics to six decimal places, leaving None and NaN blank."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        return f"{value:.6f}"

    def _write_row(self, writer, *, iteration, num_labeled, val_per_class, val_scores,
                   test_per_class, test_scores, selected) -> None:
        row = {"iteration": iteration, "num_labeled_samples": num_labeled}
        for split, per_class, (map_value, aulc) in (
            ("val", val_per_class, val_scores),
            ("test", test_per_class, test_scores),
        ):
            row[f"{split}_mAP"] = self._fmt(map_value)
            row[f"{split}_AULC"] = self._fmt(aulc)
            for cls, value in zip(self.data.class_names, per_class):
                row[f"{split}_mAP_{cls}"] = self._fmt(float(value))
        row["queried_idxes_in_latest_iteration"] = json.dumps(selected)
        writer.writerow(row)

    def run(self) -> Path:
        """Run the experiment and return the seed directory, skipping completed seeds and clearing and rerunning incomplete ones.

        Raises:
            ValueError: The existing configuration differs from the current parameters.
            RuntimeError: The strategy returns an invalid selection.
        """
        check_or_write_config(self.config_dir, self.config)
        if results_csv_complete(self.results_csv, self.max_iter):
            return self.seed_dir
        if self.seed_dir.exists():
            shutil.rmtree(self.seed_dir)
        self.seed_dir.mkdir(parents=True)

        # Query network initialization and inference, training initialization, per-epoch training and validation, and testing consume random numbers in sequence.
        # The first round has no query inference, and changing the above order affects subsequent weights and selections.
        self._seed_everything()
        data = self.data
        labeled: list[int] = []
        prev_state = None
        num_labeled_hist: list[int] = []
        val_map_hist: list[float] = []
        test_map_hist: list[float] = []

        with self.results_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._csv_fieldnames())
            writer.writeheader()
            pbar = tqdm(
                range(self.max_iter),
                desc=f"[{self.dataset}/{self.strategy_name}/seed={self.seed}]",
                unit="iter",
                position=self.pbar_position,
                leave=False,
            )
            for iteration in pbar:
                selected = self._select_samples(labeled, prev_state)
                self._assert_selection(selected, labeled)
                labeled = sorted(set(labeled) | set(selected))

                net, val_ckpt = build_training_net(
                    num_classes=data.num_classes,
                    hidden_features=self.hidden_features,
                    num_labeled=len(labeled),
                    x_val=data.val_embedding,
                    y_val=data.val_label,
                    val_valid_lengths=data.val_valid_lengths,
                    lr=self.lr,
                    warmup_ratio=self.warmup_ratio,
                    num_epochs=self.num_epochs,
                    batch_size=self.batch_size,
                    infer_batch_size=self.infer_batch_size,
                    device=self.device,
                )
                net.fit(data.train_embedding[labeled], data.train_label[labeled])
                net.module_.load_state_dict(val_ckpt.best_state)
                test_per_class, test_map = frame_wise_map(
                    net.predict_frame_proba(data.test_embedding),
                    data.test_label,
                    data.test_valid_lengths,
                )

                val_map = val_ckpt.best_map
                num_labeled_hist.append(len(labeled))
                val_map_hist.append(val_map)
                test_map_hist.append(test_map)
                val_aulc = cumulative_aulc(num_labeled_hist, val_map_hist)
                test_aulc = cumulative_aulc(num_labeled_hist, test_map_hist)

                torch.save(
                    {k: v.cpu() for k, v in val_ckpt.best_state.items()},
                    self.seed_dir / f"iter={iteration}-labeled={len(labeled)}-val_mAP={val_map:.6f}.pth",
                )
                self._write_row(
                    writer, iteration=iteration, num_labeled=len(labeled),
                    val_per_class=val_ckpt.best_per_class, val_scores=(val_map, val_aulc),
                    test_per_class=test_per_class, test_scores=(test_map, test_aulc),
                    selected=selected,
                )
                fh.flush()
                prev_state = val_ckpt.best_state

                pbar.set_postfix({"val_mAP": f"{val_map:.4f}", "test_mAP": f"{test_map:.4f}"})
        return self.seed_dir
