"""Export the sample selection comparison of seven strategies for Figure 2, caching UMAP and mismatch fields in OUTPUT_ROOT/figures/cache/."""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", dotenv=True, pythonpath=True)

# Limit thread counts before importing numpy to prevent numba/sklearn from crashing on multicore nodes without a GPU.
os.environ.setdefault("OMP_NUM_THREADS", "16")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from src.analysis.results import (  # noqa: E402
    CONFIG_FILENAME,
    DATASET_NAMES,
    REFERENCE_STRATEGY,
    STRATEGY_COLORS,
    STRATEGY_DISPLAY_NAMES,
    STRATEGY_ORDER,
    discover_config_dirs,
    load_selections,
    parse_config_filter,
)
from src.analysis.selection import (  # noqa: E402
    build_grid,
    compute_mismatch_fields,
    compute_umap_2d,
    density_alpha,
    grid_scalar_field,
    load_or_compute_array,
)
from src.analysis.style import apply_nature_style, mm, save_figure  # noqa: E402

TOP_ROW = ("random", "ft", "fl")
BOTTOM_ROW = ("mf-ft", "mf-fl", "mw-ft", "mw-fl")

POOL_COLOR = "#DDDDDD"
NEW_EDGE_COVERAGE = "#111111"
NEW_EDGE_MISMATCH = "#12E0E8"
SECOND_ROW_Y_SHIFT = -0.030        # Shift the bottom-row panels down to leave room for group labels
GROUP_LABEL_Y = 1.16
GROUP_LABEL_FONT_SIZE = 8.8
BOTTOM_TITLE_FONT_SIZE = 6.6
BOTTOM_LEGEND_X_SHIFT = -0.095


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="DataSED", choices=DATASET_NAMES, help="dataset to plot (default DataSED)")
    parser.add_argument("--keyframe", type=int, default=3,
                        help="round (0-based) of new selections to show, must be >= 1, default 3 i.e. labeled 75→100")
    parser.add_argument("--seed", type=int, default=0, help="which seed's run to show (default 0)")
    parser.add_argument("--config-hash", nargs="+", default=None, metavar="PREFIX",
                        help="keep only config_hash dirs with these name prefixes (per-strategy hash, several allowed)")
    parser.add_argument("--config-filter", nargs="+", default=[], metavar="KEY=VALUE",
                        help="keep only config_hash dirs whose config.json fields equal these values")
    parser.add_argument("--device", default=None,
                        help="mismatch-field inference device, defaults to the run's config.json training device")
    parser.add_argument("--formats", nargs="+", default=["svg", "pdf", "png"], help="export formats")
    parser.add_argument("--cache-root", default=os.environ.get("CACHE_ROOT"),
                        help="frame-level cache root (default CACHE_ROOT in .env), read only on precomputed cache miss")
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT"),
                        help="experiment output root (default OUTPUT_ROOT in .env), figures and cache go to figures/")
    args = parser.parse_args(argv)
    if args.keyframe < 1:
        parser.error("--keyframe must be >= 1: round 0 is the cold start, there is no mismatch field at selection time")
    return args


def _style_panel(ax, extent, title: str, color: str, *, ours: bool, title_fontsize: float = 8.2) -> None:
    """Set axis limits and titles, and thicken the borders for the reference strategy."""
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.6 if ours else 0.7)
        spine.set_color(color if ours else "#B8B8B8")
    ax.set_title(f"{title}  (ours)" if ours else title, fontsize=title_fontsize, fontweight="bold",
                 color=color, pad=2.5)


def _shift_axes_y(ax, dy: float) -> None:
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])


def render_selection_figure(
    umap_2d: np.ndarray,
    selections: dict[str, tuple[list[np.ndarray], list[np.ndarray]]],
    mismatch_fields: dict[str, np.ndarray],
    keyframe: int,
):
    """Plot a comparison of strategy selections from the same round on UMAP.

    Args:
        umap_2d: Coordinates of the entire pool, ``[N, 2]``.
        selections: ``{strategy: (cumulative, new)}`` for the seven strategies.
        mismatch_fields: ``{strategy: [max_iter, N]}`` mismatch fields for the four strategies in the bottom row.
        keyframe: Round to display, at least 1, overlaid on the mismatch field from round keyframe-1.

    Returns:
        matplotlib Figure.
    """
    grid = build_grid(umap_2d)
    extent = grid.extent
    alpha_mask = density_alpha(grid)
    cumulative = {s: selections[s][0][keyframe] for s in STRATEGY_ORDER}
    new = {s: selections[s][1][keyframe] for s in STRATEGY_ORDER}
    num_after = len(cumulative[TOP_ROW[0]])
    num_before = num_after - len(new[TOP_ROW[0]])
    step_label = f"${num_before}{{\\rightarrow}}{num_after}$"

    grid_fields = {s: grid_scalar_field(grid, mismatch_fields[s][keyframe - 1].astype(np.float32))
                   for s in BOTTOM_ROW}
    vmax = max(1.0, *(float(np.percentile(f, 99.0)) for f in grid_fields.values()))

    fig = plt.figure(figsize=(mm(181), mm(84)))
    gs = GridSpec(2, 4, figure=fig, wspace=0.05, hspace=0.26, left=0.008, right=0.992, top=0.930, bottom=0.100)

    # The top row shows coverage.
    for col, strategy in enumerate(TOP_ROW):
        ax = fig.add_subplot(gs[0, col])
        color = STRATEGY_COLORS[strategy]
        ax.scatter(umap_2d[:, 0], umap_2d[:, 1], s=1.0, c=POOL_COLOR, linewidths=0, rasterized=True)
        so_far, batch = cumulative[strategy], new[strategy]
        ax.scatter(umap_2d[so_far, 0], umap_2d[so_far, 1], s=6.0, c=color, alpha=0.85, linewidths=0, zorder=3)
        ax.scatter(umap_2d[batch, 0], umap_2d[batch, 1], s=30, facecolor=color,
                   edgecolor=NEW_EDGE_COVERAGE, linewidths=0.9, zorder=5)
        _style_panel(ax, extent, STRATEGY_DISPLAY_NAMES[strategy], color, ours=False)

    # The upper-right legend explains the three marker types in the top row.
    legend_ax = fig.add_subplot(gs[0, 3])
    legend_ax.axis("off")
    x_title, x_mark, x_text = 0.03, 0.10, 0.19

    def legend_row(y: float, label: str, *, facecolor, size: float, edgecolor="none", linewidth: float = 0.0):
        legend_ax.scatter([x_mark], [y], s=size, facecolor=facecolor, edgecolor=edgecolor, linewidths=linewidth,
                          transform=legend_ax.transAxes, clip_on=False, zorder=5)
        legend_ax.text(x_text, y, label, fontsize=6.6, va="center", transform=legend_ax.transAxes)

    legend_ax.text(x_title, 0.90, "Top row: coverage view", fontsize=7.0, fontweight="bold", va="top",
                   transform=legend_ax.transAxes)
    legend_row(0.70, "unlabeled pool", facecolor=POOL_COLOR, size=12)
    legend_row(0.55, f"selected so far ($n{{=}}{num_before}$)", facecolor="#555555", size=14)
    legend_row(0.40, f"newly selected ({step_label})", facecolor="#555555", size=30,
               edgecolor=NEW_EDGE_COVERAGE, linewidth=0.9)

    # The bottom row shows mismatch fields.
    bottom_axes = {}
    for col, strategy in enumerate(BOTTOM_ROW):
        ax = fig.add_subplot(gs[1, col])
        _shift_axes_y(ax, SECOND_ROW_Y_SHIFT)
        bottom_axes[strategy] = ax
        ax.set_facecolor("black")
        ax.imshow(grid_fields[strategy], extent=extent, origin="lower", cmap="magma", vmin=0, vmax=vmax,
                  alpha=alpha_mask, aspect="auto", interpolation="bilinear")
        batch = new[strategy]
        ax.scatter(umap_2d[batch, 0], umap_2d[batch, 1], s=15, facecolor="white",
                   edgecolor=NEW_EDGE_MISMATCH, linewidths=0.6, zorder=5)
        _style_panel(ax, extent, STRATEGY_DISPLAY_NAMES[strategy], STRATEGY_COLORS[strategy],
                     ours=strategy == REFERENCE_STRATEGY, title_fontsize=BOTTOM_TITLE_FONT_SIZE)
    # Group labels are centered on the right edge of the left panel and span both panels.
    for anchor, group_label in (("mf-ft", "mismatch-first (hard gating)"), ("mw-ft", "mismatch-weighted (soft)")):
        bottom_axes[anchor].annotate(group_label, xy=(1.0, GROUP_LABEL_Y), xycoords="axes fraction",
                                     ha="center", va="bottom", fontsize=GROUP_LABEL_FONT_SIZE,
                                     fontweight="bold", color="#333333")

    # The bottom legend explains the markers in the bottom row.
    strip = fig.add_axes([0.008, 0.006, 0.984, 0.058])
    strip.axis("off")
    strip.text(0.20 + BOTTOM_LEGEND_X_SHIFT, 0.55, "Bottom row: disagreement view", fontsize=7.0,
               fontweight="bold", ha="left", va="center", transform=strip.transAxes, color="#333333")
    strip.scatter([0.475 + BOTTOM_LEGEND_X_SHIFT], [0.55], s=30, facecolor="white", edgecolor=NEW_EDGE_MISMATCH,
                  linewidths=1.1, transform=strip.transAxes, clip_on=False, zorder=5)
    strip.text(0.49 + BOTTOM_LEGEND_X_SHIFT, 0.55, f"newly selected ({step_label})", fontsize=6.6,
               va="center", transform=strip.transAxes)
    colorbar_ax = strip.inset_axes([0.68 + BOTTOM_LEGEND_X_SHIFT, 0.42, 0.10, 0.28])
    mappable = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(vmin=0, vmax=vmax))
    colorbar = fig.colorbar(mappable, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks(np.arange(0, min(2, int(np.floor(vmax))) + 1))
    colorbar.ax.tick_params(labelsize=5.8, length=2, width=0.6)
    colorbar.outline.set_linewidth(0.6)
    strip.text(0.795 + BOTTOM_LEGEND_X_SHIFT, 0.55, "mismatch at selection time", fontsize=6.6,
               va="center", transform=strip.transAxes)
    return fig


def main(argv=None) -> None:
    args = parse_args(argv)
    apply_nature_style(font_size=8)
    output_root = Path(args.output_root)
    dataset, seed = args.dataset, args.seed
    config_dirs = discover_config_dirs(
        output_root, [dataset], STRATEGY_ORDER,
        config_hashes=args.config_hash, config_filter=parse_config_filter(args.config_filter),
    )
    if missing := [s for s in STRATEGY_ORDER if (dataset, s) not in config_dirs]:
        raise SystemExit(f"{output_root / dataset} is missing experiment outputs for strategies: {', '.join(missing)}")

    @functools.cache
    def frame_data():
        """Load frame-level arrays only when the precomputed cache is missing."""
        from src.data.arrays import load_frame_data

        return load_frame_data(Path(args.cache_root) / dataset)

    precompute_dir = output_root / "figures" / "cache"
    umap_2d = load_or_compute_array(precompute_dir / f"{dataset}_umap2d.npy",
                                    lambda: compute_umap_2d(frame_data().seg_embedding))
    selections = {s: load_selections(config_dirs[(dataset, s)], seed) for s in STRATEGY_ORDER}
    mismatch_fields = {}
    for strategy in BOTTOM_ROW:
        cfg_dir = config_dirs[(dataset, strategy)]
        config = json.loads((cfg_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))
        cumulative = selections[strategy][0]
        mismatch_fields[strategy] = load_or_compute_array(
            precompute_dir / f"{dataset}_{strategy}_{cfg_dir.name}_seed{seed}_mismatch.npy",
            lambda: compute_mismatch_fields(
                frame_data(), cfg_dir / f"seed_{seed}", cumulative,
                hidden_features=config["hidden_features"],
                batch_size=config["batch_size"],
                infer_batch_size=config["infer_batch_size"],
                device=args.device or config["device"],
            ),
        )

    fig = render_selection_figure(umap_2d, selections, mismatch_fields, args.keyframe)
    stem = f"fig2_selection_{dataset}_seed{seed}_iter{args.keyframe}"
    for path in save_figure(fig, stem, output_root / "figures", args.formats):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
