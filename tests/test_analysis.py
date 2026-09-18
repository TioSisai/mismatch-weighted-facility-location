"""Test summaries and plotting with mock outputs and synthetic caches."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rootutils

from src.analysis.results import (
    CURVE_METRICS,
    STRATEGY_ORDER,
    aggregate_mean_band,
    aulc_summary,
    discover_config_dirs,
    final_aulc,
    load_curves,
    load_selections,
    paired_wilcoxon,
    parse_config_filter,
)
from src.analysis.selection import (
    GRID_K,
    GRID_RES,
    MARGIN,
    SmoothingGrid,
    build_grid,
    compute_mismatch_fields,
    density_alpha,
    grid_scalar_field,
    load_or_compute_array,
)

_ROOT = rootutils.find_root(__file__, indicator=".project-root")
_CLASSES = ("c0", "c1")
_STEP = 5


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(output_root: Path, dataset: str, strategy: str, seed: int, test_map, val_map=None, *,
               config_hash: str = "aaaaaaaaaaaa", selections=None, **config_fields) -> Path:
    """Write the configuration and single-seed results in learner format and return the seed directory."""
    test_map = np.asarray(test_map, dtype=np.float64)
    val_map = test_map - 0.01 if val_map is None else np.asarray(val_map, dtype=np.float64)
    cfg_dir = output_root / dataset / strategy / config_hash
    cfg_dir.mkdir(parents=True, exist_ok=True)
    config = {"dataset": dataset, "strategy": strategy, "max_iter": len(test_map), "step_size": _STEP,
              "device": "cpu", **config_fields}
    (cfg_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    seed_dir = cfg_dir / f"seed_{seed}"
    seed_dir.mkdir()
    fieldnames = ["iteration", "num_labeled_samples", "val_mAP", "val_AULC", "test_mAP", "test_AULC",
                  *[f"val_mAP_{c}" for c in _CLASSES], *[f"test_mAP_{c}" for c in _CLASSES],
                  "queried_idxes_in_latest_iteration"]
    xs = _STEP * np.arange(1, len(test_map) + 1)

    def aulc(ys, i):
        return "" if i == 0 else f"{np.trapezoid(ys[:i + 1], xs[:i + 1]) / (xs[i] - xs[0]):.6f}"

    with (seed_dir / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(test_map)):
            row = {
                "iteration": i, "num_labeled_samples": xs[i],
                "val_mAP": f"{val_map[i]:.6f}", "val_AULC": aulc(val_map, i),
                "test_mAP": f"{test_map[i]:.6f}", "test_AULC": aulc(test_map, i),
                "queried_idxes_in_latest_iteration": json.dumps(
                    selections[i] if selections is not None else list(range(i * _STEP, (i + 1) * _STEP))),
            }
            row |= {f"{split}_mAP_{c}": "0.500000" for split in ("val", "test") for c in _CLASSES}
            writer.writerow(row)
    return seed_dir


def _write_tree(output_root: Path, datasets=("DESED", "DataSED"), num_seeds: int = 6, max_iter: int = 4) -> None:
    """Construct complete experiment outputs with curves increasing in STRATEGY_ORDER."""
    rng = np.random.default_rng(0)
    for dataset in datasets:
        for rank, strategy in enumerate(STRATEGY_ORDER):
            for seed in range(num_seeds):
                curve = 0.3 + 0.05 * rank + 0.02 * np.arange(max_iter) + rng.uniform(0, 0.01, max_iter)
                _write_run(output_root, dataset, strategy, seed, curve)


def _curves_from_final(final_values: dict[str, np.ndarray], dataset: str = "DESED") -> pd.DataFrame:
    """Construct a two-round long-format table from each strategy's final-round AULC, with NaN AULC in the first round."""
    rows = []
    for strategy, values in final_values.items():
        for seed, value in enumerate(values):
            for iteration, aulc in ((0, np.nan), (1, value)):
                rows.append({"dataset": dataset, "strategy": strategy, "seed": seed, "iteration": iteration,
                             "num_labeled": _STEP * (iteration + 1), "val_mAP": 0.5, "test_mAP": 0.5,
                             "val_AULC": aulc, "test_AULC": aulc})
    curves = pd.DataFrame(rows)
    curves["strategy"] = pd.Categorical(curves["strategy"], categories=list(STRATEGY_ORDER), ordered=True)
    return curves


def test_strategy_names_match_registry():
    from src.strategies import registry

    assert STRATEGY_ORDER == registry.STRATEGY_NAMES


def test_parse_config_filter_keeps_json_types():
    assert parse_config_filter(["num_epochs=25", "lr=1e-3", "device=cuda", "flag=true"]) == {
        "num_epochs": 25, "lr": 1e-3, "device": "cuda", "flag": True}
    with pytest.raises(ValueError):
        parse_config_filter(["num_epochs"])


def test_discover_requires_unique_hash_and_supports_filters(tmp_path):
    _write_run(tmp_path, "DESED", "fl", 0, [0.5, 0.6], config_hash="111111111111", num_epochs=25)
    _write_run(tmp_path, "DESED", "fl", 0, [0.5, 0.6], config_hash="222222222222", num_epochs=5)
    _write_run(tmp_path, "DESED", "mw-fl", 0, [0.5, 0.6], config_hash="333333333333", num_epochs=25)

    with pytest.raises(ValueError, match=r"DESED/fl(.|\n)*111111111111: num_epochs=25"):
        discover_config_dirs(tmp_path)

    by_hash = discover_config_dirs(tmp_path, config_hashes=["1111", "3333"])
    assert by_hash == {("DESED", "fl"): tmp_path / "DESED/fl/111111111111",
                       ("DESED", "mw-fl"): tmp_path / "DESED/mw-fl/333333333333"}
    by_field = discover_config_dirs(tmp_path, config_filter={"num_epochs": 5})
    assert by_field == {("DESED", "fl"): tmp_path / "DESED/fl/222222222222"}


def test_load_curves_schema_order_and_incomplete_seed(tmp_path):
    _write_run(tmp_path, "DESED", "mw-fl", 1, [0.5, 0.6, 0.7])
    _write_run(tmp_path, "DESED", "mw-fl", 0, [0.4, 0.5, 0.6])
    _write_run(tmp_path, "DESED", "random", 0, [0.1, 0.2, 0.3])
    incomplete = _write_run(tmp_path, "DESED", "random", 1, [0.1, 0.2, 0.3])
    rows = (incomplete / "results.csv").read_text().splitlines()
    (incomplete / "results.csv").write_text("\n".join(rows[:-1]) + "\n")

    with pytest.warns(UserWarning, match="seed_1"):
        curves = load_curves(discover_config_dirs(tmp_path))
    assert list(curves.columns) == ["dataset", "strategy", "seed", "iteration", "num_labeled", *CURVE_METRICS]
    assert list(zip(curves.strategy, curves.seed)) == [("random", 0)] * 3 + [("mw-fl", 0)] * 3 + [("mw-fl", 1)] * 3
    assert curves.num_labeled.tolist()[:3] == [5, 10, 15]
    assert curves.test_AULC.isna().tolist() == [True, False, False] * 3
    # Trapezoidal integration over three equally spaced points (0.1/2 + 0.2 + 0.3/2) / 2 = 0.2
    assert curves.test_AULC.iloc[2] == pytest.approx(0.2)


def test_load_selections_accumulates_in_query_order(tmp_path):
    config_dir = _write_run(tmp_path, "DESED", "ft", 0, [0.1, 0.2], selections=[[7, 3], [1, 9]]).parent
    cumulative, new = load_selections(config_dir, 0)
    assert [c.tolist() for c in cumulative] == [[7, 3], [7, 3, 1, 9]]
    assert [n.tolist() for n in new] == [[7, 3], [1, 9]]


def test_final_aulc_summary_and_rank():
    curves = _curves_from_final({"ft": np.array([0.5, 0.6, 0.7]), "fl": np.array([0.8, 0.8, 0.9]),
                                 "random": np.array([0.1, 0.2, 0.3])})
    final = final_aulc(curves, "test_AULC")
    assert final.test_AULC.tolist() == [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9]

    summary = aulc_summary(curves, "test_AULC").set_index("strategy")
    assert summary.loc["fl", "mean"] == pytest.approx(np.mean([0.8, 0.8, 0.9]))
    assert summary.loc["ft", "std"] == pytest.approx(np.std([0.5, 0.6, 0.7], ddof=1))
    assert summary["rank"].to_dict() == {"random": 3, "ft": 2, "fl": 1}
    assert summary["n_seeds"].tolist() == [3, 3, 3]


def test_paired_wilcoxon_exact_two_sided():
    base = np.linspace(0.5, 0.6, 10)
    gaps = np.arange(1, 11) * 1e-3
    curves = _curves_from_final({
        "mw-fl": base + gaps,
        "fl": base,
        "ft": base + gaps * np.where(np.arange(10) == 0, 2, 0),  # ft is higher by 1e-3 for seed 0
    })
    table = paired_wilcoxon(curves, "test_AULC").set_index("strategy")
    assert table.index.tolist() == ["ft", "fl"]
    assert table.loc["fl", "wins"] == 10
    assert table.loc["fl", "n_pairs"] == 10
    assert table.loc["fl", "mean_diff"] == pytest.approx(gaps.mean())
    assert table.loc["fl", "p_value"] == pytest.approx(2 / 2**10)
    assert table.loc["fl", "statistic"] == 0
    assert table.loc["ft", "wins"] == 9
    assert table.loc["ft", "p_value"] == pytest.approx(4 / 2**10)


def test_aggregate_mean_band():
    curves = _curves_from_final({"fl": np.array([0.2, 0.4, 0.9])})
    x, mean, lo, hi, n = aggregate_mean_band(curves, "DESED", "fl", "test_AULC")
    assert x.tolist() == [5, 10]
    assert np.isnan(mean[0]) and n.tolist() == [0, 3]
    assert (mean[1], lo[1], hi[1]) == (pytest.approx(0.5), 0.2, 0.9)


def test_summarize_writes_marked_tables(tmp_path, capsys):
    _write_tree(tmp_path)
    _load_script("summarize").main(["--output-root", str(tmp_path)])

    tables = tmp_path / "tables"
    assert sorted(p.name for p in tables.iterdir()) == sorted(
        f"{d}_{kind}.{ext}" for d in ("DESED", "DataSED") for kind in ("aulc", "wilcoxon") for ext in ("md", "csv"))
    aulc_md = (tables / "DESED_aulc.md").read_text()
    # Mock curves increase in strategy order, with mw-fl best and mw-ft second best.
    assert "| MW-FL | FL | soft weight | **" in aulc_md
    assert "| MW-FT | FT | soft weight | <u>" in aulc_md
    aulc_csv = pd.read_csv(tables / "DESED_aulc.csv")
    assert aulc_csv.strategy.tolist() == ["random", "ft", "mf-ft", "mw-ft", "fl", "mf-fl", "mw-fl"]
    assert aulc_csv.set_index("strategy").loc["mw-fl", ["test_AULC_rank", "val_AULC_rank"]].tolist() == [1, 1]

    wilcoxon_csv = pd.read_csv(tables / "DataSED_wilcoxon.csv")
    assert len(wilcoxon_csv) == 12 and "mw-fl" not in set(wilcoxon_csv.strategy)
    assert (wilcoxon_csv.wins == 6).all()
    assert wilcoxon_csv.p_value.tolist() == pytest.approx([2 / 2**6] * 12)
    assert "MW-FL vs Random | test_AULC | 6/6 |" in capsys.readouterr().out


def test_plot_curves_fallback_legend(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    plot_curves = _load_script("plot_curves")
    monkeypatch.setattr(plot_curves.shutil, "which", lambda name: None)
    _write_tree(tmp_path)
    plot_curves.main(["--output-root", str(tmp_path), "--formats", "png"])
    assert sorted(p.name for p in (tmp_path / "figures").iterdir()) == sorted([
        "fig1_mAP_vs_labeled_TEST_2panel.png", "figS1_mAP_vs_labeled_VAL_2panel.png",
        "fig1_mAP_TEST_DESED.png", "fig1_mAP_TEST_DataSED.png",
        "figS1_mAP_VAL_DESED.png", "figS1_mAP_VAL_DataSED.png",
    ])

    curves = load_curves(discover_config_dirs(tmp_path))
    fig, ax = plt.subplots()
    plot_curves.add_aulc_legend(ax, aulc_summary(curves, "test_AULC"), "DESED", usetex=False, fontsize=5.6)
    texts = ax.get_legend().get_texts()
    labels = [t.get_text() for t in texts]
    assert len(labels) == 8 and labels[-1].startswith("MW-FL (ours)")
    assert len({label.index("AULC") if i == 0 else label.index("±") - 5 for i, label in enumerate(labels)}) == 1
    styles = {label.split()[0]: (t.get_fontweight(), t.get_fontstyle()) for label, t in zip(labels, texts)}
    assert styles["MW-FL"] == ("bold", "normal")
    assert styles["MW-FT"] == ("normal", "italic")
    assert styles["FL"] == ("normal", "normal")
    plt.close(fig)


def _manual_grid(num_points: int = 30, seed: int = 0) -> SmoothingGrid:
    rng = np.random.default_rng(seed)
    num_cells = GRID_RES * GRID_RES
    dist = np.sort(rng.uniform(0.01, 2.0, (num_cells, GRID_K)), axis=1).astype(np.float32)
    idx = rng.integers(num_points, size=(num_cells, GRID_K)).astype(np.int32)
    return SmoothingGrid(extent=(0.0, 1.0, 0.0, 1.0), idx=idx, dist=dist)


def test_grid_scalar_field_weighted_mean():
    grid = _manual_grid()
    values = np.arange(30, dtype=np.int64)
    field = grid_scalar_field(grid, values)
    assert field.shape == (GRID_RES, GRID_RES) and field.dtype == np.float32
    weights = 1.0 / (grid.dist[123].astype(np.float64) + 1e-6)
    expected = (weights * values[grid.idx[123]]).sum() / weights.sum()
    assert field.ravel()[123] == pytest.approx(expected, rel=1e-5)
    assert np.allclose(grid_scalar_field(grid, np.full(30, 2)), 2.0)


def test_density_alpha_range_and_thresholds():
    grid = _manual_grid()
    alpha = density_alpha(grid).ravel()
    nearest = grid.dist[:, 0]
    d_in, d_out = np.percentile(nearest, 48), np.percentile(nearest, 84)
    assert density_alpha(grid).shape == (GRID_RES, GRID_RES)
    assert alpha.min() == 0.0 and alpha.max() == 1.0
    assert (alpha[nearest <= d_in] == 1.0).all() and (alpha[nearest >= d_out] == 0.0).all()


@pytest.mark.parametrize("use_cuml", [True, False])
def test_build_grid_matches_brute_force(monkeypatch, use_cuml):
    from src.strategies import backends

    if use_cuml and not backends.HAS_CUML:
        pytest.skip("cuml is unavailable")
    monkeypatch.setattr(backends, "HAS_CUML", use_cuml)
    umap_2d = np.random.default_rng(1).normal(size=(50, 2)).astype(np.float32)
    grid = build_grid(umap_2d)
    assert grid.extent == pytest.approx((umap_2d[:, 0].min() - MARGIN, umap_2d[:, 0].max() + MARGIN,
                                         umap_2d[:, 1].min() - MARGIN, umap_2d[:, 1].max() + MARGIN))
    assert grid.idx.shape == grid.dist.shape == (GRID_RES * GRID_RES, GRID_K)
    assert (np.diff(grid.dist, axis=1) >= -1e-5).all()
    xs = np.linspace(grid.extent[0], grid.extent[1], GRID_RES, dtype=np.float32)
    ys = np.linspace(grid.extent[2], grid.extent[3], GRID_RES, dtype=np.float32)
    cells = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)[::997]
    brute = np.linalg.norm(cells[:, None, :] - umap_2d[None], axis=-1).min(axis=1)
    assert grid.dist[::997, 0] == pytest.approx(brute, abs=1e-4)


def test_load_or_compute_array_caches(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return np.arange(4)

    path = tmp_path / "sub" / "x.npy"
    assert load_or_compute_array(path, compute).tolist() == [0, 1, 2, 3]
    assert load_or_compute_array(path, compute).tolist() == [0, 1, 2, 3]
    assert len(calls) == 1 and sorted(p.name for p in path.parent.iterdir()) == ["x.npy"]


def _write_fake_checkpoints(run_dir: Path, num_classes: int, cumulative, hidden_features: int = 8) -> None:
    """Write scaled-up random weights for each round using learner naming."""
    import torch

    from src.models.classifier import FrameMLPClassifier

    run_dir.mkdir(parents=True, exist_ok=True)
    for iteration, labeled in enumerate(cumulative):
        torch.manual_seed(100 + iteration)
        module = FrameMLPClassifier(num_classes, hidden_features=hidden_features).initialize().module_
        # Scale up weights to move predictions away from 0.5 and increase differences in mismatch scores.
        state = {k: v * 20 for k, v in module.state_dict().items()}
        torch.save(state, run_dir / f"iter={iteration}-labeled={len(labeled)}-val_mAP=0.500000.pth")


def test_compute_mismatch_fields_matches_query_time(synthetic_cache, tmp_path):
    import torch

    from src.data.arrays import load_frame_data
    from src.data.frames import segment_max_pool
    from src.models.classifier import build_inference_net
    from src.strategies.mismatch import candidate_mismatch

    data = load_frame_data(synthetic_cache)
    order = np.random.default_rng(0).permutation(data.n_train)
    cumulative = [order[:4 * (i + 1)] for i in range(3)]
    run_dir = tmp_path / "seed_0"
    _write_fake_checkpoints(run_dir, data.num_classes, cumulative)

    fields = compute_mismatch_fields(data, run_dir, cumulative, hidden_features=8, batch_size=4,
                                     infer_batch_size=16, device="cpu")
    assert fields.shape == (3, data.n_train)
    for iteration, labeled in enumerate(cumulative):
        (ckpt,) = run_dir.glob(f"iter={iteration}-*.pth")
        net = build_inference_net(num_classes=data.num_classes, hidden_features=8, state_dict=torch.load(ckpt),
                                  batch_size=4, infer_batch_size=16, device="cpu")
        seg_proba = segment_max_pool(net.predict_frame_proba(data.train_embedding), data.train_valid_lengths)
        unlabeled = np.setdiff1d(np.arange(data.n_train), labeled)
        expected = candidate_mismatch(data.seg_embedding, unlabeled, np.sort(labeled), seg_proba, data.seg_labels)
        assert np.array_equal(fields[iteration, unlabeled], expected)
        hamming = np.abs((seg_proba[labeled] >= 0.5).astype(int) - data.seg_labels[labeled]).sum(axis=1)
        assert np.array_equal(fields[iteration, labeled], hamming)
    assert fields.max() > 0


def test_plot_selection_end_to_end(synthetic_cache, tmp_path):
    from src.data.arrays import read_array_meta

    plot_selection = _load_script("plot_selection")
    output_root = tmp_path / "outputs"
    n_train, num_classes, _ = read_array_meta(synthetic_cache)
    order = np.random.default_rng(0).permutation(n_train)
    selections = [order[4 * i:4 * (i + 1)].tolist() for i in range(3)]
    for strategy in STRATEGY_ORDER:
        seed_dir = _write_run(output_root, "DataSED", strategy, 0, [0.1, 0.2, 0.3], selections=selections,
                              hidden_features=8, batch_size=4, infer_batch_size=16)
        if strategy in plot_selection.BOTTOM_ROW:
            _write_fake_checkpoints(seed_dir, num_classes, [order[:4 * (i + 1)] for i in range(3)])

    argv = ["--keyframe", "2", "--formats", "png", "--cache-root", str(synthetic_cache.parent),
            "--output-root", str(output_root)]
    # Link the Toy cache as DataSED for the plotting script.
    (synthetic_cache.parent / "DataSED").symlink_to(synthetic_cache)
    plot_selection.main(argv)
    cache_files = sorted(p.name for p in (output_root / "figures" / "cache").iterdir())
    assert cache_files == sorted(["DataSED_umap2d.npy"] + [
        f"DataSED_{s}_aaaaaaaaaaaa_seed0_mismatch.npy" for s in plot_selection.BOTTOM_ROW])
    assert np.load(output_root / "figures" / "cache" / "DataSED_mw-fl_aaaaaaaaaaaa_seed0_mismatch.npy").shape \
        == (3, n_train)
    figure = output_root / "figures" / "fig2_selection_DataSED_seed0_iter2.png"
    assert figure.is_file()

    figure.unlink()
    plot_selection.main([*argv, "--cache-root", str(tmp_path / "missing")])  # Precomputed caches are complete, so the frame-level cache is not needed.
    assert figure.is_file()

    with pytest.raises(SystemExit):
        plot_selection.parse_args(["--keyframe", "0"])
