"""Test serial and multiprocessing pipelines with small CPU experiments."""

from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from conftest import build_synthetic_frame_cache
from src.learner import OVERALL_FIELDNAMES

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pipeline.py"

# The training pool has 40 segments, enough for two rounds selecting 4 each.
MAX_ITER = 2
STEP_SIZE = 4
SMALL_ARGS = [
    "--max-iter", str(MAX_ITER), "--step-size", str(STEP_SIZE),
    "--hidden-features", "32", "--num-epochs", "2", "--batch-size", "4",
    "--infer-batch-size", "8", "--device", "cpu",
]


@pytest.fixture(scope="module")
def pipeline():
    spec = importlib.util.spec_from_file_location("pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cache_root(tmp_path) -> Path:
    """Build a synthetic cache at tmp_path/cache/DESED and return the cache root directory."""
    build_synthetic_frame_cache(tmp_path / "cache" / "DESED")
    return tmp_path / "cache"


def _root_args(cache_root: Path, output_root: Path) -> list[str]:
    return ["--cache-root", str(cache_root), "--output-root", str(output_root)]


def _results_csv(output_root: Path, strategy: str, seed: int) -> Path:
    (cfg_dir,) = (output_root / "DESED" / strategy).iterdir()
    return cfg_dir / f"seed_{seed}" / "results.csv"


def test_parse_args_defaults_follow_paper_protocol(pipeline, monkeypatch):
    monkeypatch.setenv("CACHE_ROOT", "/cache-from-env")
    monkeypatch.setenv("OUTPUT_ROOT", "/output-from-env")
    args = pipeline.parse_args([])
    assert args.datasets == ["DESED", "DataSED"]
    assert args.strategies == ["random", "ft", "fl", "mf-ft", "mf-fl", "mw-ft", "mw-fl"]
    assert (args.seed, args.runs, args.max_iter, args.step_size) == (0, 10, 20, 25)
    assert (args.hidden_features, args.lr, args.warmup_ratio) == (512, 1e-3, 0.2)
    assert (args.num_epochs, args.batch_size, args.infer_batch_size) == (25, 16, 128)
    assert (args.device, args.n_tasks) == ("auto", 16)
    assert (args.cache_root, args.output_root) == ("/cache-from-env", "/output-from-env")


@pytest.mark.parametrize("argv", [
    ["--strategies", "mwfl"],
    ["--datasets", "ESC-50"],
    ["--device", "cuda:0"],
])
def test_parse_args_rejects_unknown_choices(pipeline, argv):
    with pytest.raises(SystemExit):
        pipeline.parse_args(argv)


def test_main_requires_roots(pipeline, monkeypatch):
    monkeypatch.delenv("CACHE_ROOT", raising=False)
    monkeypatch.delenv("OUTPUT_ROOT", raising=False)
    with pytest.raises(SystemExit):
        pipeline.main([])


def test_prepare_tasks_expands_grid(pipeline, cache_root, tmp_path):
    output_root = tmp_path / "out"
    args = pipeline.parse_args([
        "--datasets", "DESED", "--strategies", "random", "mw-fl", "--seed", "3",
        "--runs", "2", *SMALL_ARGS, *_root_args(cache_root, output_root),
    ])
    tasks, n_skipped = pipeline.prepare_tasks(args, cache_root, output_root, "cpu")

    assert n_skipped == 0
    assert [(t["learner"]["strategy_name"], t["learner"]["seed"]) for t in tasks] == [
        ("random", 3), ("random", 4), ("mw-fl", 3), ("mw-fl", 4),
    ]
    assert {t["cache_dir"] for t in tasks} == {str(cache_root / "DESED")}
    assert pickle.loads(pickle.dumps(tasks)) == tasks
    learner_params = set(inspect.signature(pipeline.ActiveLearner).parameters)
    assert set(tasks[0]["learner"]) | {"data", "pbar_position"} == learner_params
    for strategy in ("random", "mw-fl"):
        (config_path,) = (output_root / "DESED" / strategy).glob("*/config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert (config["strategy"], config["device"]) == (strategy, "cpu")
        assert (config["n_train"], config["num_classes"]) == (40, 3)


def test_prepare_tasks_skips_only_complete_seeds(pipeline, cache_root, tmp_path):
    output_root = tmp_path / "out"
    args = pipeline.parse_args([
        "--datasets", "DESED", "--strategies", "random", "--runs", "2",
        *SMALL_ARGS, *_root_args(cache_root, output_root),
    ])
    pipeline.prepare_tasks(args, cache_root, output_root, "cpu")
    for seed, num_rows in ((0, MAX_ITER), (1, MAX_ITER - 1)):
        results_csv = _results_csv(output_root, "random", seed)
        results_csv.parent.mkdir(parents=True)
        with results_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(OVERALL_FIELDNAMES)
            writer.writerows([[0] * len(OVERALL_FIELDNAMES)] * num_rows)

    tasks, n_skipped = pipeline.prepare_tasks(args, cache_root, output_root, "cpu")
    assert n_skipped == 1
    assert [t["learner"]["seed"] for t in tasks] == [1]


def test_prepare_tasks_fails_fast(pipeline, cache_root, tmp_path):
    output_root = tmp_path / "out"
    args = pipeline.parse_args([
        "--datasets", "DESED", "--strategies", "random", "--runs", "1",
        *SMALL_ARGS, *_root_args(cache_root, output_root),
    ])
    pipeline.prepare_tasks(args, cache_root, output_root, "cpu")
    (config_path,) = (output_root / "DESED" / "random").glob("*/config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_path.write_text(json.dumps({**config, "lr": 0.5}), encoding="utf-8")
    with pytest.raises(ValueError):
        pipeline.prepare_tasks(args, cache_root, output_root, "cpu")

    with pytest.raises(SystemExit):
        pipeline.prepare_tasks(args, tmp_path / "missing", output_root, "cpu")


def test_prepare_tasks_warns_on_foreign_backend_cache(pipeline, cache_root, tmp_path, capsys):
    output_root = tmp_path / "out"
    args = pipeline.parse_args([
        "--datasets", "DESED", "--strategies", "random", "--runs", "1",
        *SMALL_ARGS, *_root_args(cache_root, output_root),
    ])
    meta_path = cache_root / "DESED" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps({**meta, "backend": "other+backend"}), encoding="utf-8")
    tasks, _ = pipeline.prepare_tasks(args, cache_root, output_root, "cpu")
    assert len(tasks) == 1
    assert "other+backend" in capsys.readouterr().err


def test_main_serial_end_to_end(pipeline, cache_root, tmp_path, monkeypatch, capsys):
    output_root = tmp_path / "out"
    argv = [
        "--datasets", "DESED", "--strategies", "random", "mw-fl", "--runs", "1",
        "--n-tasks", "1", *SMALL_ARGS, *_root_args(cache_root, output_root),
    ]
    pipeline.main(argv)

    for strategy in ("random", "mw-fl"):
        results_csv = _results_csv(output_root, strategy, seed=0)
        assert pipeline.results_csv_complete(results_csv, MAX_ITER)
        with results_csv.open() as fh:
            rows = list(csv.DictReader(fh))
        assert [int(row["num_labeled_samples"]) for row in rows] == [4, 8]
        selected = [json.loads(row["queried_idxes_in_latest_iteration"]) for row in rows]
        assert all(len(batch) == STEP_SIZE for batch in selected)
        assert len(set(selected[0]) | set(selected[1])) == MAX_ITER * STEP_SIZE
        assert len(list(results_csv.parent.glob("iter=*-labeled=*-val_mAP=*.pth"))) == MAX_ITER

    def fail_if_constructed(**kwargs):
        raise AssertionError("a completed seed must not run again")

    monkeypatch.setattr(pipeline, "ActiveLearner", fail_if_constructed)
    capsys.readouterr()
    pipeline.main(argv)
    assert "2 completed and skipped, 0 to run" in capsys.readouterr().out


def _seed_files(output_root: Path) -> list[Path]:
    """Return sorted relative file paths within each seed directory."""
    return sorted(p.relative_to(output_root) for p in output_root.rglob("seed_*/*") if p.is_file())


def test_parallel_matches_serial(cache_root, tmp_path):
    common = [
        "--datasets", "DESED", "--strategies", "mf-ft", "mw-fl", "--runs", "2",
        *SMALL_ARGS, "--cache-root", str(cache_root),
    ]
    # Subprocesses launch as scripts without inheriting OMP_NUM_THREADS, matching the actual runtime environment.
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    for mode, n_tasks in (("serial", "1"), ("parallel", "2")):
        subprocess.run(
            [sys.executable, str(SCRIPT), *common, "--n-tasks", n_tasks,
             "--output-root", str(tmp_path / mode)],
            check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    serial_root, parallel_root = tmp_path / "serial", tmp_path / "parallel"
    relative_paths = _seed_files(serial_root)
    assert relative_paths == _seed_files(parallel_root)
    assert len(relative_paths) == 2 * 2 * (MAX_ITER + 1)  # Strategies × seeds × (per-round weights + CSV)
    for relative in relative_paths:
        if relative.suffix == ".csv":
            assert (serial_root / relative).read_bytes() == (parallel_root / relative).read_bytes()
            continue
        serial_state = torch.load(serial_root / relative)
        parallel_state = torch.load(parallel_root / relative)
        assert serial_state.keys() == parallel_state.keys()
        assert all(torch.equal(serial_state[k], parallel_state[k]) for k in serial_state), relative
