"""Before/after training-throughput benchmark on one B200.

Runs the production 214M step shape for a few dozen steps per variant, in one
container, against two source trees: the code before the throughput work
(``before``, a ``git archive`` of ``--before-commit``) and the current working
tree (``after``). Data paths: a fixed on-device batch (compute ceiling), the Hub
stream with on-the-fly tokenization (the accepted run's path), and pre-tokenized
shards. The run card is ``docs/run-cards/throughput-bench-b200.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from esme_pretrain.launch.gpu_smoke import _check_spend_cap, _reserved_cost_usd
from esme_pretrain.launch.modal_artifacts import local_git_commit, local_git_dirty
from esme_pretrain.launch.modal_image import build_pretrain_modal_image
from esme_pretrain.pretrain_run import load_pretrain_config

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_GPU = "B200"
BENCH_VOLUME = "esme-pretrain-bench"
BENCH_TIMEOUT_SECONDS = 75 * 60
# Six variants at this timeout fit inside the function timeout, so a stalled
# variant costs its own slot and never the rows already written.
VARIANT_TIMEOUT_SECONDS = 11 * 60
VARIANT_SCRIPT = REPO_ROOT / "scripts" / "throughput_bench_variant.py"
DEFAULT_BEFORE_COMMIT = "5cd1851"
DEFAULT_OUTPUT_DIR = Path("runs/throughput-bench-b200")
# Gitignored; launch() extracts the before tree here before the app runs.
BEFORE_TREE_DIR = REPO_ROOT / "runs" / "throughput-bench-b200" / "before-tree"
RESULTS_ROOT = Path("/bench/results")

# (variant, source tree, data path). The repeated before-fixed run at the end checks
# that the GPU did not drift (thermal or neighbor noise) across the session.
VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("before-fixed", "before", "fixed"),
    ("after-fixed", "after", "fixed"),
    ("before-stream", "before", "stream"),
    ("after-stream", "after", "stream"),
    ("after-shards", "after", "shards"),
    ("before-fixed-repeat", "before", "fixed"),
)
SOURCE_TREES = {"before": "/root/src_before", "after": "/root/src"}

try:
    import modal
except ImportError:  # pragma: no cover - Modal is supplied by the dev extra.
    modal = None

if modal is not None:  # pragma: no cover - exercised on Modal, not in unit tests.
    # Module-level, so Modal imports run_bench by reference in the container. A
    # pickled-by-value function would carry local-interpreter bytecode into the
    # image's Python and fail to load.
    image = (
        build_pretrain_modal_image(repo_root_src=str(REPO_ROOT / "src"), modal_module=modal)
        .add_local_dir(str(BEFORE_TREE_DIR / "src"), remote_path=SOURCE_TREES["before"])
        .add_local_file(str(VARIANT_SCRIPT), remote_path="/root/throughput_bench_variant.py")
    )
    app = modal.App("esme-pretrain-throughput-bench", image=image)
    bench_volume = modal.Volume.from_name(BENCH_VOLUME)

    @app.function(gpu=BENCH_GPU, timeout=BENCH_TIMEOUT_SECONDS, volumes={"/bench": bench_volume})
    def run_bench(config_payload: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return run_bench_body(config_payload, params, commit_results=bench_volume.commit)


def launch(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the before/after B200 throughput benchmark.")
    parser.add_argument("--config", type=Path, default=Path("configs/pretrain_214m_b200.json"))
    parser.add_argument("--before-commit", default=DEFAULT_BEFORE_COMMIT)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=40)
    parser.add_argument("--spend-cap-usd", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--approved", action="store_true", help="Required to rent the GPU.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate config, commits, and the spend cap without launching (no spend).",
    )
    args = parser.parse_args(argv)

    try:
        if args.warmup_steps < 1 or args.measure_steps < 5:
            raise ValueError("--warmup-steps must be >= 1 and --measure-steps >= 5")
        config = load_pretrain_config(args.config)
        profile = config.payload["runtime"]["gpu_profiles"][BENCH_GPU]
        worst_case_cost = _reserved_cost_usd(
            timeout_seconds=BENCH_TIMEOUT_SECONDS, usd_per_hour=float(profile["usd_per_hour"])
        )
        _check_spend_cap(worst_case_cost, args.spend_cap_usd)
        before_commit = _resolve_commit(args.before_commit)
    except (ValueError, KeyError) as error:
        print(f"throughput bench refused: {error}", file=sys.stderr)
        return 2

    plan = {
        "gpu": BENCH_GPU,
        "before_commit": before_commit,
        "after_commit": local_git_commit(REPO_ROOT),
        "after_dirty": local_git_dirty(REPO_ROOT),
        "variants": [variant for variant, _tree, _data in VARIANTS],
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "timeout_hours": BENCH_TIMEOUT_SECONDS / 3600,
        "worst_case_cost_usd": worst_case_cost,
        "spend_cap_usd": args.spend_cap_usd,
        "volume": BENCH_VOLUME,
    }
    if args.check_only:
        print(json.dumps({"status": "check_ok", **plan}, indent=2, sort_keys=True))
        return 0
    if not args.approved:
        print("throughput bench refused: pass --approved to rent the GPU", file=sys.stderr)
        return 2
    if plan["after_dirty"]:
        print("throughput bench refused: commit the working tree first", file=sys.stderr)
        return 2
    if modal is None:
        print("throughput bench failed: modal is not installed", file=sys.stderr)
        return 2

    if BEFORE_TREE_DIR.exists():
        shutil.rmtree(BEFORE_TREE_DIR)
    _extract_source_tree(before_commit, BEFORE_TREE_DIR)
    run_id = f"{plan['after_commit'][:7]}-{time.strftime('%Y%m%dT%H%M%S')}"
    params = {
        "run_id": run_id,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
    }
    print(
        f"throughput bench: run {run_id}; each variant's row is committed to "
        f"{BENCH_VOLUME}:/results/{run_id}/ as it finishes",
        flush=True,
    )
    wall_start = time.perf_counter()
    # Detached: a client disconnect does not stop the paid run, and the rows on the
    # volume survive it.
    with modal.enable_output(), app.run(detach=True):
        result = run_bench.remote(config.payload, params)
    result = {**plan, **result, "client_wall_seconds": time.perf_counter() - wall_start}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"throughput-bench-{plan['after_commit'][:7]}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({**result, "local_output_path": str(output_path)}, indent=2, sort_keys=True))
    failed = [row["variant"] for row in result["variants_run"] if row["status"] != "ok"]
    if failed:
        print(f"throughput bench: variants did not complete: {failed}", file=sys.stderr)
        return 1
    return 0


def _resolve_commit(ref: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"--before-commit {ref!r} is not a commit")
    return result.stdout.strip()


def _extract_source_tree(commit: str, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "archive", "--format=tar", commit, "src"],
        check=True,
        capture_output=True,
    )
    with tarfile.open(fileobj=BytesIO(archive.stdout)) as tar:
        tar.extractall(destination, filter="data")


def run_bench_body(
    config_payload: dict[str, Any], params: dict[str, Any], *, commit_results: Any
) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path("/tmp/bench-config.json")
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    results_dir = RESULTS_ROOT / params["run_id"]
    results_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for variant, tree, data in VARIANTS:
        row = _run_variant(variant, tree, data, config_path=config_path, params=params)
        (results_dir / f"{variant}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
        )
        commit_results()
        rows.append(row)
    return {"variants_run": rows, "remote_wall_seconds": time.perf_counter() - started}


def _run_variant(
    variant: str, tree: str, data: str, *, config_path: Path, params: dict[str, Any]
) -> dict[str, Any]:
    # -P keeps the script's directory (/root) off sys.path. Modal mounts this module's
    # package at /root/esme_pretrain, which would otherwise shadow PYTHONPATH and run
    # every variant on the same tree.
    command = [
        sys.executable,
        "-P",
        "/root/throughput_bench_variant.py",
        "--expected-source",
        SOURCE_TREES[tree],
        "--variant",
        variant,
        "--data",
        data,
        "--config",
        str(config_path),
        "--tokenizer",
        "/bench/tokenizer.json",
        "--shard-manifest",
        "/bench/shards/train-manifest.json",
        "--warmup-steps",
        str(params["warmup_steps"]),
        "--measure-steps",
        str(params["measure_steps"]),
        "--output-dir",
        f"/tmp/bench-{variant}",
    ]
    env = {**os.environ, "PYTHONPATH": SOURCE_TREES[tree]}
    print(f"throughput bench: {variant} begin", flush=True)
    try:
        completed = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=VARIANT_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        return {"variant": variant, "status": "timeout", "timeout_seconds": VARIANT_TIMEOUT_SECONDS}
    results = [
        line.removeprefix("BENCH_RESULT ")
        for line in completed.stdout.splitlines()
        if line.startswith("BENCH_RESULT ")
    ]
    if completed.returncode != 0 or len(results) != 1:
        return {
            "variant": variant,
            "status": "failed",
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-2000:],
        }
    print(f"throughput bench: {variant} done", flush=True)
    return {"status": "ok", **json.loads(results[0])}
