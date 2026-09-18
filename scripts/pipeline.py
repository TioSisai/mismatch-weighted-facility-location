"""Run active learning experiments across datasets, strategies, and seeds in batches, with multiprocessing and resumption by seed."""

from __future__ import annotations

import argparse
import atexit
import ctypes
import functools
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", dotenv=True, pythonpath=True)

# Limit thread counts before importing torch to avoid CPU contention among parallel processes.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from src.data import DATASET_NAMES  # noqa: E402
from src.data.arrays import (  # noqa: E402
    META_FILENAME,
    FrameData,
    load_frame_data,
    read_array_meta,
)
from src.learner import (  # noqa: E402
    ActiveLearner,
    check_or_write_config,
    config_dir,
    results_csv_complete,
    strategy_config,
)
from src.strategies import STRATEGY_NAMES  # noqa: E402
from src.strategies.backends import backend_signature  # noqa: E402

MPS_CONTROL = "nvidia-cuda-mps-control"

# Load prctl during the single-threaded phase to avoid calling dlopen in preexec_fn after fork.
_PRCTL = ctypes.CDLL(None).prctl
_PR_SET_PDEATHSIG = 1

# Progress row for a worker process; reserve 0 for the total progress bar.
_WORKER_SLOT: int = 1
# Allow tasks already queued but not yet running to be aborted as well.
_ABORT_EVENT = None


@functools.lru_cache(maxsize=None)
def _load_frame_data(cache_dir: str) -> FrameData:
    """Reuse mmap arrays by directory within a process."""
    return load_frame_data(Path(cache_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse experiment arguments, defaulting cache and output paths to CACHE_ROOT and OUTPUT_ROOT."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES),
                        choices=list(DATASET_NAMES), help="subset of datasets to run (default all)")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGY_NAMES),
                        choices=STRATEGY_NAMES, help="subset of query strategies to run (default all)")
    parser.add_argument("--seed", type=int, default=0,
                        help="base random seed (paper protocol: 0)")
    parser.add_argument("--runs", type=int, default=10,
                        help="number of runs, using seeds seed..seed+runs-1 in order (paper protocol: 10)")
    parser.add_argument("--max-iter", type=int, default=20,
                        help="total number of active learning rounds (paper protocol: 20)")
    parser.add_argument("--step-size", type=int, default=25,
                        help="number of segments selected per round (paper protocol: 25)")
    parser.add_argument("--hidden-features", type=int, default=512,
                        help="hidden dimension of the frame-level MLP classification head (paper protocol: 512)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="peak learning rate (paper protocol: 1e-3)")
    parser.add_argument("--warmup-ratio", type=float, default=0.2,
                        help="ratio of total steps for linear warmup, the rest cosine annealing (paper protocol: 0.2)")
    parser.add_argument("--num-epochs", type=int, default=25,
                        help="number of epochs per round when training from scratch (paper protocol: 25)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="training batch size (paper protocol: 16)")
    parser.add_argument("--infer-batch-size", type=int, default=128,
                        help="batch size for val/test/query inference (paper protocol: 128)")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                        help="torch device, auto uses cuda when CUDA is available, otherwise cpu")
    parser.add_argument("--n-tasks", type=int, default=16,
                        help="number of parallel tasks, 1 means serial")
    parser.add_argument("--cache-root", default=os.environ.get("CACHE_ROOT"),
                        help="frame-level cache root directory, containing {dataset}/ (default CACHE_ROOT in .env)")
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT"),
                        help="experiment output root directory (default OUTPUT_ROOT in .env)")
    return parser.parse_args(argv)


def prepare_tasks(
    args: argparse.Namespace, cache_root: Path, output_root: Path, device: str
) -> tuple[list[dict], int]:
    """Expand experiment combinations, write configurations, and skip completed seeds.

    Args:
        args: Return value of parse_args.
        cache_root: Root directory for the data cache.
        output_root: Root output directory.
        device: Resolved cpu or cuda device to write to the configuration.

    Returns:
        ``(tasks, n_skipped)``. Each task is a serializable ``{"cache_dir": str, "learner": dict}``,
        where learner contains ActiveLearner arguments except data and pbar_position.

    Raises:
        SystemExit: The data cache is missing.
        ValueError: The existing configuration differs from the current arguments.
    """
    tasks, n_skipped = [], 0
    for dataset in args.datasets:
        cache_dir = cache_root / dataset
        if not (cache_dir / META_FILENAME).is_file():
            raise SystemExit(f"frame-level cache not found: {cache_dir} (download it from Zenodo, see README)")
        n_train, num_classes, cached_backend = read_array_meta(cache_dir)
        # Rounding differences between the cache and the current backend affect sample selection.
        if cached_backend != backend_signature():
            print(f"warning: the distance matrix of {dataset} was computed with the {cached_backend} backend, "
                  f"the current backend is {backend_signature()}, and the results will differ from the paper",
                  file=sys.stderr)
        for strategy in args.strategies:
            config = strategy_config(
                dataset=dataset, strategy=strategy, step_size=args.step_size,
                max_iter=args.max_iter, hidden_features=args.hidden_features,
                lr=args.lr, warmup_ratio=args.warmup_ratio,
                num_epochs=args.num_epochs, batch_size=args.batch_size,
                infer_batch_size=args.infer_batch_size, num_classes=num_classes,
                n_train=n_train, device=device,
            )
            cfg_dir = config_dir(output_root, config)
            check_or_write_config(cfg_dir, config)
            for run_index in range(args.runs):
                seed = args.seed + run_index
                if results_csv_complete(cfg_dir / f"seed_{seed}" / "results.csv", args.max_iter):
                    n_skipped += 1
                    continue
                tasks.append(dict(cache_dir=str(cache_dir), learner=dict(
                    dataset=dataset, strategy_name=strategy,
                    output_root=str(output_root), step_size=args.step_size,
                    max_iter=args.max_iter, hidden_features=args.hidden_features,
                    lr=args.lr, warmup_ratio=args.warmup_ratio,
                    num_epochs=args.num_epochs, seed=seed, device=device,
                    batch_size=args.batch_size,
                    infer_batch_size=args.infer_batch_size,
                )))
    return tasks, n_skipped


def make_total_pbar(n_tasks: int) -> tqdm:
    """Create the total progress bar at row 0, with experiment progress bars starting at row 1."""
    return tqdm(total=n_tasks, desc="total progress", unit="exp", position=0, leave=True)


def run_task(task: dict) -> None:
    """Run a single experiment task generated by prepare_tasks."""
    if _ABORT_EVENT is not None and _ABORT_EVENT.is_set():
        return
    data = _load_frame_data(task["cache_dir"])
    ActiveLearner(data=data, pbar_position=_WORKER_SLOT, **task["learner"]).run()


def _quit_mps(control: subprocess.Popen) -> None:
    """Send quit to shut down MPS, and send SIGTERM on timeout."""
    try:
        subprocess.run([MPS_CONTROL], input="quit", text=True, check=False, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        control.wait(timeout=30)
    except subprocess.TimeoutExpired:
        control.terminate()


def _ensure_mps() -> None:
    """Reuse an existing MPS pipe, or start MPS dedicated to this process and shut it down on exit."""
    if "CUDA_MPS_PIPE_DIRECTORY" in os.environ:
        print("CUDA MPS: reusing the external daemon")
        return
    # MPS socket paths are limited to about 107 bytes; use a short path isolated by PID under TMPDIR.
    mps_root = Path(os.environ.get("TMPDIR", "/tmp")) / f"mps_{os.getpid()}"
    (mps_root / "pipe").mkdir(parents=True, exist_ok=True)
    (mps_root / "log").mkdir(parents=True, exist_ok=True)
    # Workers inherit the pipe configuration and join MPS when creating CUDA contexts.
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(mps_root / "pipe")
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(mps_root / "log")
    try:
        # Run in the foreground and set PDEATHSIG so the kernel cleans up MPS when the parent exits.
        control = subprocess.Popen(
            [MPS_CONTROL, "-f"],
            # A separate session prevents Ctrl+C from terminating the control process before clients detach.
            start_new_session=True,
            preexec_fn=lambda: _PRCTL(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError:
        os.environ.pop("CUDA_MPS_PIPE_DIRECTORY")
        os.environ.pop("CUDA_MPS_LOG_DIRECTORY")
        print("CUDA MPS: daemon failed to start, this parallel run bypasses MPS (only throughput is affected)",
              file=sys.stderr)
        return
    atexit.register(_quit_mps, control)
    print("CUDA MPS: daemon started (shut down automatically on exit)")


def _mps_command(command: str, timeout: float) -> str | None:
    """Run an MPS control command and return stdout, or None on failure or timeout."""
    try:
        proc = subprocess.run([MPS_CONTROL], input=command, text=True, timeout=timeout,
                              check=True, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout


def _parse_pids(output: str | None) -> list[int]:
    """Parse output lines containing only digits as PIDs."""
    return [int(line.strip()) for line in (output or "").splitlines() if line.strip().isdigit()]


def _terminate_mps_clients(own_pids: set[int], timeout: float,
                           deadline: float) -> tuple[int, int]:
    """Detach MPS clients in own_pids and return the counts of successes and attempts."""
    done = attempted = 0
    for server in _parse_pids(_mps_command("get_server_list", timeout)):
        for client in _parse_pids(_mps_command(f"get_client_list {server}", timeout)):
            if client not in own_pids:  # Clients from other jobs may connect to the same server
                continue
            if time.monotonic() >= deadline:
                print("MPS: detach timed out, the remaining workers will be SIGTERM'd without synchronizing",
                      file=sys.stderr)
                return done, attempted
            attempted += 1
            if _mps_command(f"terminate_client {server} {client}", timeout) is not None:
                done += 1
            else:
                print(f"MPS: terminate_client {server} {client} failed, this worker will be SIGTERM'd "
                      "without synchronizing", file=sys.stderr)
    return done, attempted


def _detach_mps_clients(procs: list, budget_s: float = 15.0) -> None:
    """Detach this process's MPS workers according to NVIDIA's protocol."""
    if "CUDA_MPS_PIPE_DIRECTORY" not in os.environ or not procs:
        return
    # Only warn on detachment failure, and limit the total time to budget_s.
    done, attempted = _terminate_mps_clients(
        {proc.pid for proc in procs}, timeout=min(budget_s, 5.0),
        deadline=time.monotonic() + budget_s,
    )
    if attempted:
        print(f"MPS: {done}/{attempted} workers cleanly detached from the server per the official protocol",
              file=sys.stderr)


def _terminate_pool(pool: ProcessPoolExecutor, abort_event) -> None:
    """Set the abort flag, detach MPS clients, then send SIGTERM to workers."""
    # Ignore SIGINT during cleanup and restore it afterward to prevent repeated interrupts from disrupting cleanup.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    abort_event.set()
    print("Interrupt received, terminating all workers…", file=sys.stderr)
    procs = list((pool._processes or {}).values())  # Take a snapshot first, as the management thread modifies this concurrently
    # Detach MPS clients before sending SIGTERM, as required by NVIDIA.
    _detach_mps_clients(procs)
    for proc in procs:
        proc.terminate()
    pool.shutdown(wait=False, cancel_futures=True)
    print(f"SIGTERM sent to {len(procs)} workers", file=sys.stderr)
    signal.signal(signal.SIGINT, signal.default_int_handler)


def _pool_worker_init(tqdm_lock, slot_counter, abort_event) -> None:
    """Initialize a spawn worker, share the progress lock and abort flag, and assign a progress row."""
    global _WORKER_SLOT, _ABORT_EVENT
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # The parent process handles Ctrl+C centrally
    # Have the kernel send SIGTERM when the parent exits, and check whether the parent exited before registration.
    _PRCTL(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    if os.getppid() == 1:
        os._exit(1)
    tqdm.set_lock(tqdm_lock)
    _ABORT_EVENT = abort_event
    with slot_counter.get_lock():
        _WORKER_SLOT = slot_counter.value
        slot_counter.value += 1


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.cache_root:
        raise SystemExit("missing --cache-root (and .env provides no CACHE_ROOT)")
    if not args.output_root:
        raise SystemExit("missing --output-root (and .env provides no OUTPUT_ROOT)")
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tasks, n_skipped = prepare_tasks(args, Path(args.cache_root), Path(args.output_root), device)
    print(f"device={device} seeds={args.seed}..{args.seed + args.runs - 1} "
          f"tasks total {len(tasks) + n_skipped}: {n_skipped} completed and skipped, "
          f"{len(tasks)} to run, {args.n_tasks} in parallel")
    if not tasks:
        return
    print("progress bars: row 0 is the total progress, each row below is one running experiment")

    if args.n_tasks == 1 or len(tasks) == 1:
        with make_total_pbar(len(tasks)) as total_pbar:
            for task in tasks:
                run_task(task)
                total_pbar.update()
        return

    if device == "cuda":
        _ensure_mps()
    # spawn initializes CUDA and random streams independently in each process, keeping results consistent with serial execution.
    ctx = mp.get_context("spawn")
    abort_event = ctx.Event()
    # Share the progress lock only among workers to avoid blocking the parent if the lock holder exits unexpectedly.
    with make_total_pbar(len(tasks)) as total_pbar, ProcessPoolExecutor(
        max_workers=min(args.n_tasks, len(tasks)),
        mp_context=ctx,
        initializer=_pool_worker_init,
        initargs=(ctx.RLock(), ctx.Value("i", 1), abort_event),
    ) as pool:
        # Close the total progress bar before printing exception information.
        try:
            futures = [pool.submit(run_task, task) for task in tasks]
            for future in as_completed(futures):
                future.result()
                total_pbar.update()
        except KeyboardInterrupt:
            # Workers ignore SIGINT and are terminated by the parent.
            total_pbar.close()
            _terminate_pool(pool, abort_event)
            raise
        except BrokenProcessPool:
            # The executor has terminated the workers following the unexpected exit.
            total_pbar.close()
            print("a worker died unexpectedly; the process pool is broken (common causes: out-of-memory kill, "
                  "GPU/driver fault, check nvidia-smi and the kernel log); complete seed results already written "
                  "remain valid, rerun after fixing the environment to resume automatically.", file=sys.stderr)
            raise
        except BaseException:
            # After a task fails, skip pending tasks and wait for running tasks to finish; Ctrl+C terminates them immediately while waiting.
            total_pbar.close()
            try:
                traceback.print_exc()
                print("the first task failed, tasks not yet started are aborted; running tasks will finish "
                      "(results valid). If waiting is unnecessary (e.g. a systemic fault), press Ctrl+C to "
                      "terminate all immediately.", file=sys.stderr)
                abort_event.set()
                pool.shutdown(cancel_futures=True)
            except KeyboardInterrupt:
                _terminate_pool(pool, abort_event)
                raise
            raise


if __name__ == "__main__":
    main()
