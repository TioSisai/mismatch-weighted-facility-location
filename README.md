<!-- Cover First, Disagree Softly -->
# Cover First, Disagree Softly

<!-- Official implementation of the paper "Cover First, Disagree Softly: Rethinking Mismatch-First Active Learning for Frame-Level Audio Classification" by Shiqi Zhang and Tuomas Virtanen. -->
Official implementation of *Cover First, Disagree Softly: Rethinking Mismatch-First Active Learning for Frame-Level Audio Classification* by Shiqi Zhang and Tuomas Virtanen ([arXiv:2607.13571](https://arxiv.org/abs/2607.13571)).

<!-- The experiments compare farthest-point traversal (FT) and facility location (FL), combined with no disagreement signal, mismatch-first selection (MF), or mismatch weighting (MW), using random sampling as a baseline. -->
The experiments compare farthest traversal (FT) and facility location (FL) with no disagreement signal, mismatch-first selection (MF), or mismatch weighting (MW), alongside random sampling.

<!-- Command-line strategy names and names in the paper. mf-ft corresponds to MFFT. -->
| CLI strategy | Paper name |
|---|---|
| `random` | Random |
| `ft` | FT |
| `mf-ft` | MF-FT (MFFT) |
| `mw-ft` | MW-FT |
| `fl` | FL |
| `mf-fl` | MF-FL |
| `mw-fl` | MW-FL |

<!-- By default, 25 ten-second audio segments are selected per round, with 20 rounds and 10 random seeds, and a frame-level MLP (2048→512→C) is trained from scratch each round. In the first round, FT variants start from a random point and FL variants use uniform weights. -->
Defaults follow the paper protocol: 25 ten-second segments per round, 20 rounds, 10 seeds, and a frame-level MLP (2048→512→C) trained from scratch each round. In the first round, FT variants start from a random point and FL variants use uniform weights.

<!-- Environment setup -->
## Setup

<!-- Only Linux is supported. The reproduction environment uses CSC Roihu's NVIDIA GH200 (aarch64), CUDA 13.2, and Python 3.12.14. See requirements.txt for dependency lower bounds. The multi-GPU launcher also requires nvidia-smi and taskset. -->
Linux is required. The reproduction environment uses CSC Roihu's NVIDIA GH200 (aarch64), CUDA 13.2, and Python 3.12.14. Dependency lower bounds are in [requirements.txt](requirements.txt). The multi-GPU launcher also requires `nvidia-smi` and `taskset`.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
# Optional GPU computation backend. Package names must match the CUDA major version.
pip install -r requirements-gpu.txt --extra-index-url https://pypi.nvidia.com
```

<!-- The paper results use CUDA training and the cuml+cupy computation backend. When optional dependencies are missing, sklearn, numpy, and umap-learn are used automatically. Floating-point rounding differences may change selections near ties. Both the training device and backend are recorded in config.json. -->
Paper results use CUDA training and the `cuml+cupy` backend. Without the optional dependencies, computation falls back to sklearn, numpy, and umap-learn. Rounding differences can change selections near ties. The training device and backend are recorded in `config.json`.

<!-- Tests use synthetic data and can run on CPU with python -m pytest. -->
Tests use synthetic data and run on CPU with `python -m pytest`.

<!-- Data and paths -->
## Data and paths

The cache has been published on ZENODO with DOI [10.5281/zenodo.22825178](https://doi.org/10.5281/zenodo.22825178). Once downloaded, unpack it into `DESED/` and `DataSED/` under `CACHE_ROOT` (about 5.9 GiB total). Scripts read the precomputed arrays. Raw audio is not included.

<!-- Copy .env.example to .env and set the cache directory CACHE_ROOT and output directory OUTPUT_ROOT. Scripts load this file automatically. Command-line arguments take precedence, and ${VAR} can reference earlier variables. -->
Copy [.env.example](.env.example) to `.env` and set `CACHE_ROOT` and `OUTPUT_ROOT`. Scripts load `.env` automatically. Command-line arguments take precedence, and `${VAR}` can reference earlier variables.

<!--
Cache files, shapes and dtypes, and contents are listed below.
{split}_embedding.npy contains PANNs frame-level embeddings, including padded frames.
{split}_label.npy contains frame-level multi-label targets, and {split}_valid_lengths.npy contains the number of valid frames per segment.
train_seg_embedding.npy contains the mean embeddings over valid frames, and train_seg_labels.npy contains the maximum label values over valid frames.
pairwise_distances.npy contains the Euclidean distance matrix of training-pool segment representations.
meta.json records classes, array shapes, and the distance computation backend.
segments.csv records source audio files and segment start and end times for each split.
-->
| File | Shape and dtype | Content |
|---|---|---|
| `{split}_embedding.npy` | `[N, 31, 2048]` float32 | PANNs frame embeddings, including padding |
| `{split}_label.npy` | `[N, 31, C]` float32 | Frame-level multi-hot labels |
| `{split}_valid_lengths.npy` | `[N]` int64 | Valid frames per segment, `1..31` |
| `train_seg_embedding.npy` | `[N, 2048]` float32 | Mean embedding over valid frames |
| `train_seg_labels.npy` | `[N, C]` int64 | Max-pooled labels over valid frames |
| `pairwise_distances.npy` | `[N, N]` float64 | Train-pool Euclidean distances |
| `meta.json` | JSON | Classes, array shapes, distance backend |
| `segments.csv` | CSV | Source audio and `[seg_start, seg_end)` per split |

<!-- {split} is train, val, or test. Array rows within each split follow the order in segments.csv, and selection indices correspond to training-pool row numbers. -->
`{split}` is `train`, `val`, or `test`. Array rows follow the corresponding split in `segments.csv`. Selection indices refer to train-pool rows.

<!-- Cache provenance -->
### Cache provenance

<!-- Audio is divided into non-overlapping 10-second segments, retaining short trailing segments with more than zero valid frames. Segments are resampled to 32 kHz and encoded into 31×2048 frame embeddings by PANNs Cnn14_DecisionLevelMax on CUDA with a batch size of 32. -->
Audio was divided into non-overlapping 10-second segments, retaining short tails with at least one valid frame. Segments were resampled to 32 kHz and encoded into 31×2048 frame embeddings by PANNs `Cnn14_DecisionLevelMax` ([weights](https://zenodo.org/records/3987831)), using CUDA and batch size 32.

<!-- DESED uses DCASE 2021 Task 4's synthetic21_train, synthetic21_validation, and eval/public, containing 10000, 2500, and 693 segments, respectively. The 10 classes are the sorted union of event_label across the three annotation files. -->
[DESED](https://github.com/turpaultn/DESED) uses DCASE 2021 Task 4's `synthetic21_train`, `synthetic21_validation`, and `eval/public` (10000/2500/693 segments). The 10 classes are the sorted union of `event_label` across the three annotation files.

<!-- DataSED uses the original paper's 712 recordings S-0001 through S-0712, totaling 6808 segments and 22 classes. Annotations come primarily from Polyphonic_sound_detection.csv, supplemented by Monophonic_sound_detection.csv for S-0704.wav through S-0712.wav, adding the Wind turbine class. The extra S-0713.wav through S-0717.wav are excluded because they fall outside the original paper's scope and lack polyphonic annotations. Each segment's rarest class is used as its stratification label, with seed 0 and an 8:1:1 split yielding 5446/681/681 training, validation, and test segments. -->
[DataSED](https://doi.org/10.5281/zenodo.15346092) uses the original paper's 712 recordings `S-0001` through `S-0712`, yielding 6808 segments and 22 classes. Annotations come from `Polyphonic_sound_detection.csv`, supplemented by `Monophonic_sound_detection.csv` for `S-0704.wav` through `S-0712.wav`, which adds the Wind turbine class. The extra `S-0713.wav` through `S-0717.wav` are excluded because they fall outside the paper's scope and lack polyphonic annotations. Stratification uses each segment's globally rarest label, with seed 0 and an 8:1:1 train/val/test split (5446/681/681 segments).

<!-- Run experiments -->
## Run experiments

<!-- Run the following command from the repository root to cover 2 datasets, 7 strategies, and 10 seeds. -->
Run from the repository root to cover 2 datasets, 7 strategies, and 10 seeds.

```bash
python scripts/pipeline.py --n-tasks 16
```

<!-- --datasets and --strategies select subsets, --seed is the starting seed, and --runs is the number of consecutive seeds. --n-tasks controls concurrency, with 1 running serially. Processes share arrays through mmap and use independent random streams. Parallel CUDA runs automatically use MPS when available. OMP_NUM_THREADS defaults to 4 per process and can be overridden through the environment. -->
Use `--datasets` and `--strategies` for subsets, and `--seed` and `--runs` for consecutive seeds. `--n-tasks 1` runs serially. Workers share arrays through mmap and keep independent random streams. CUDA parallel runs use MPS when available. Each process defaults to `OMP_NUM_THREADS=4`, overridable through the environment.

<!-- For multiple GPUs, the launcher binds each process to the specified GPU and its NUMA-local CPU cores and forwards the remaining arguments to pipeline.py. By default, it uses python from PATH. MWFL_PYTHON can specify the interpreter. The split below covers all 140 experiments. -->
For multiple GPUs, the launcher binds each process to its GPU and NUMA-local CPU cores and forwards other arguments to `pipeline.py`. Set `MWFL_PYTHON` to override `python` from PATH. This split covers all 140 experiments.

```bash
# Launch each command in a separate terminal or job step.
scripts/run_pipeline.sh --cuda-id 0 --datasets DESED   --seed 0 --runs 5
scripts/run_pipeline.sh --cuda-id 1 --datasets DESED   --seed 5 --runs 5
scripts/run_pipeline.sh --cuda-id 2 --datasets DataSED --seed 0 --runs 5
scripts/run_pipeline.sh --cuda-id 3 --datasets DataSED --seed 5 --runs 5
```

<!-- Dataset, strategy, and seed combinations should be disjoint across processes to avoid overwriting the same output. When multiple jobs share a GPU, control total concurrency to avoid exceeding the MPS client limit. -->
Keep dataset/strategy/seed combinations disjoint to avoid overwriting outputs. When jobs share a GPU, keep total concurrency within the MPS client limit.

<!-- Outputs and resuming -->
### Outputs and resuming

```
{OUTPUT_ROOT}/{dataset}/{strategy}/{config_hash}/
├── config.json
└── seed_{s}/
    ├── results.csv
    └── iter={i}-labeled={n}-val_mAP={m}.pth
```

<!-- config_hash is the first 12 digits of the configuration's SHA-256 hash. Changed configurations write to a new directory. Each checkpoint stores the state_dict from the epoch with the best validation mAP in that round, with tensors on CPU. -->
`config_hash` is the first 12 SHA-256 digits of the configuration, so changed settings use a new directory. Each checkpoint stores the round's best-val-mAP `state_dict` as CPU tensors.

<!-- results.csv has one row per round, recording the round index starting from 0, cumulative labeled count, validation and test mAP, per-class AP, normalized trapezoidal AULC, and the JSON index list queried_idxes_in_latest_iteration preserving selection order. Values use six decimal places. Undefined values, such as first-round AULC and AP for classes without positive examples, are left blank. -->
`results.csv` records the round index (from 0), labeled count, val/test mAP, per-class AP, normalized trapezoidal AULC, and `queried_idxes_in_latest_iteration` as a JSON list in selection order. Values use six decimal places. Undefined values, including first-round AULC and AP for classes without positive frames, are blank.

<!-- Rerunning the same command skips completed seeds, deletes incomplete seeds' directories, and restarts them from round 0. The cache and all checkpoints for the default experiments occupy about 17 GB. -->
Rerunning the command skips complete seeds and deletes and restarts incomplete seeds from round 0. The cache and all default-protocol checkpoints occupy about 17 GB.

<!-- Summaries and plots -->
## Tables and figures

```bash
python scripts/summarize.py
```

<!-- Only the final AULC of complete seeds is summarized, producing aulc and wilcoxon tables (Markdown and CSV) for each dataset under OUTPUT_ROOT/tables/. The former contains means, standard deviations, and ranks. The latter uses exact two-sided Wilcoxon tests paired by seed, with mw-fl as the default reference strategy, which can be changed with --reference. -->
Complete seeds' final AULC values produce `{dataset}_aulc.{md,csv}` and `{dataset}_wilcoxon.{md,csv}` under `{OUTPUT_ROOT}/tables/`. Tables report mean, standard deviation, rank, and exact two-sided Wilcoxon tests paired by seed. The reference strategy defaults to `mw-fl` (`--reference`).

<!-- When multiple configuration directories exist for the same dataset and strategy, filter them with --config-hash PREFIX ... or --config-filter device=cuda backend=cuml+cupy. -->
If a dataset/strategy has multiple configurations, select one with `--config-hash PREFIX ...` or `--config-filter device=cuda backend=cuml+cupy`.

```bash
python scripts/plot_curves.py
python scripts/plot_selection.py
```

<!-- Figures are saved as SVG, PDF, and PNG under OUTPUT_ROOT/figures/. plot_curves.py produces Fig. 1 (test set) and Fig. S1 (validation set). plot_selection.py produces Fig. 2, using DataSED, seed 0, and round 3 by default. UMAP and disagreement fields are cached in figures/cache/. -->
Figures are saved as SVG, PDF, and PNG under `{OUTPUT_ROOT}/figures/`. `plot_curves.py` produces Fig. 1 (test) and Fig. S1 (val). `plot_selection.py` produces Fig. 2, defaulting to DataSED, seed 0, round 3, and caches UMAP and mismatch fields in `{OUTPUT_ROOT}/figures/cache/`.

<!-- Reproduced results -->
## Reproduced results

<!-- Test AULC means ± standard deviations (10 seeds) match the paper at the precision shown below. The paper used CSC Mahti's A100 (x86_64), while this repository uses CSC Roihu's GH200 (aarch64), the same cache, and default arguments. -->
Test AULC (mean ± std, 10 seeds) matches the paper at the precision shown. The paper used CSC Mahti's A100 (x86_64), and this repository used CSC Roihu's GH200 (aarch64), the same cache, and default arguments.

<!-- Strategies and their test AULC on DESED and DataSED. -->
| Strategy | DESED | DataSED |
|---|---|---|
| Random | 0.715 ± 0.013 | 0.602 ± 0.018 |
| FT | 0.708 ± 0.009 | 0.651 ± 0.005 |
| MF-FT | 0.704 ± 0.011 | 0.633 ± 0.014 |
| MW-FT | 0.707 ± 0.011 | 0.650 ± 0.006 |
| FL | 0.731 ± 0.000 | 0.657 ± 0.001 |
| MF-FL | 0.689 ± 0.010 | 0.618 ± 0.008 |
| MW-FL | **0.735 ± 0.003** | **0.660 ± 0.002** |

<!-- On both datasets, MW-FL's test AULC differs significantly from each of the other six strategies, with p ≤ 0.006 in exact two-sided paired Wilcoxon tests. -->
MW-FL outperforms all six alternatives on both datasets (paired exact two-sided Wilcoxon test, p ≤ 0.006).

<!-- Citation and license -->
## Citation and license

```bibtex
@article{zhang2026cover,
  title   = {Cover First, Disagree Softly: Rethinking Mismatch-First Active Learning for Frame-Level Audio Classification},
  author  = {Zhang, Shiqi and Virtanen, Tuomas},
  journal = {arXiv preprint arXiv:2607.13571},
  year    = {2026}
}
```

Code is released under the [MIT license](LICENSE). The `DESED/` and `DataSED/` caches use CC BY 4.0 and CC BY-NC-SA 4.0, respectively. See [cache licenses and sources](CACHE_LICENSES.md) for attribution and scope.
