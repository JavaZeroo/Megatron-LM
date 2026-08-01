#!/usr/bin/env python3
"""Reproduce Megatron-LM PR #5960's unfused THD CP>1 parity failure.

This launcher reuses the repository's end-to-end CP2-vs-CP1 parity test.  It
only overrides test dispatch so that H100/H200 or supported Blackwell GPUs
execute the unfused DSA path; production loss code and parity assertions are
not modified.  The CP layout helper itself still requires its CuTe kernel, so
this full-path reproduction cannot run on A100.

Example:

    CUDA_VISIBLE_DEVICES=0,1 python3.12 \
        tools/debug/pr5960/repro_cp2_unfused.py --loss-mode both

The launcher starts two torchrun workers itself.  On PR head 3ae4e2bb the
ratio-4 indexer cases are expected to fail CP2-vs-CP1 gradient parity.  The
same command should pass after the CP loss teacher includes sliding-window
mass.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

EXPECTED_PR_HEAD = "3ae4e2bb49fee19b6cefa1d562af438e9340c5a9"
TEST_RELATIVE_PATH = Path(
    "tests/unit_tests/transformer/experimental_attention_variant/"
    "test_dsv4_hybrid_attention_cp.py"
)
TARGET_TEST = (
    f"{TEST_RELATIVE_PATH}::TestDSv4HybridAttentionTHDCP::"
    "test_thd_cp_matches_full_reference_forward_backward"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real Megatron-LM THD CP2-vs-CP1 test while forcing the "
            "unfused DSA and unfused RoPE paths."
        )
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("MEGATRON_REPO", str(Path(__file__).resolve().parents[3]))),
        help="Megatron-LM checkout containing PR #5960 (or a candidate fix).",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("sparse", "dense", "both"),
        default="both",
        help="Indexer-loss branch to exercise (default: both).",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _validate_checkout(repo: Path) -> None:
    if not repo.is_dir():
        raise SystemExit(f"Megatron-LM checkout does not exist: {repo}")
    test_path = repo / TEST_RELATIVE_PATH
    if not test_path.is_file():
        raise SystemExit(f"Required CP parity test does not exist: {test_path}")
    if sys.version_info < (3, 12):
        raise SystemExit(
            f"Python >= 3.12 is required by this checkout; got {sys.version.split()[0]}"
        )


def _launch_two_workers(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    _validate_checkout(repo)
    head = _git_head(repo)
    print(f"[PR5960-CP2] repo={repo}", flush=True)
    print(f"[PR5960-CP2] head={head}", flush=True)
    if head != EXPECTED_PR_HEAD:
        print(
            f"[PR5960-CP2] NOTE: current audited PR head is {EXPECTED_PR_HEAD}; "
            "a different head is useful for before/after comparison but may produce "
            "a different result.",
            flush=True,
        )
    print(
        "[PR5960-CP2] topology=TP1/PP1/CP2, layout=THD, "
        "dsa_kernel_backend=none, apply_rope_fusion=false",
        flush=True,
    )
    print(
        "[PR5960-CP2] RED on the audited PR reproduces the bug; GREEN after the fix "
        "is the regression gate.",
        flush=True,
    )

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(Path(__file__).resolve()),
        "--worker",
        "--repo",
        str(repo),
        "--loss-mode",
        args.loss_mode,
    ]
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return subprocess.call(command, cwd=repo, env=env)


class _ForceUnfusedPlugin:
    """Force the existing parity test onto the exact path under review."""

    def __init__(self) -> None:
        self.patched = False
        self.diagnostics_patched = False
        self.passed_calls = 0
        self.failed_calls = 0
        self.skipped_setups = 0

    def pytest_collection_modifyitems(self, session, config, items) -> None:
        del session, config
        for item in items:
            module = item.module
            module_file = Path(getattr(module, "__file__", ""))
            if module_file.name == TEST_RELATIVE_PATH.name:
                # This test normally auto-selects fused DSA on H100.  Returning
                # False makes its config set dsa_kernel_backend="none" on every GPU.
                module._dsv4_cp_fused_kernels_available = lambda: False
                self.patched = True
                if not self.diagnostics_patched:
                    original_assert = module._assert_cp_tensor_match

                    def assert_with_immediate_diagnostics(actual, expected, label):
                        try:
                            return original_assert(actual, expected, label)
                        except AssertionError as exc:
                            print(f"[PR5960-CP2] ASSERTION: {exc}", flush=True)
                            raise

                    module._assert_cp_tensor_match = assert_with_immediate_diagnostics
                    self.diagnostics_patched = True

        # These CP tests do not consume /opt/data.  Avoid an unrelated download
        # attempt in the repository-wide autouse fixture when running standalone.
        conftest = sys.modules.get("tests.unit_tests.conftest")
        if conftest is not None:
            conftest.download_and_extract_asset = lambda *args, **kwargs: None

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call":
            self.passed_calls += int(report.passed)
            self.failed_calls += int(report.failed)
        elif report.when == "setup" and report.skipped:
            self.skipped_setups += 1


def _selection_expression(loss_mode: str) -> str:
    common = "cp2 and unfused_rope"
    if loss_mode == "sparse":
        return f"{common} and ratio_4_indexer_sparse"
    if loss_mode == "dense":
        return f"{common} and ratio_4_indexer_dense"
    return f"{common} and (ratio_4_indexer_sparse or ratio_4_indexer_dense)"


def _run_worker(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    _validate_checkout(repo)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 2:
        raise SystemExit(
            f"This reproduction requires exactly two workers; got WORLD_SIZE={world_size}"
        )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    try:
        import pytest
    except ImportError as exc:
        raise SystemExit(
            "pytest is unavailable. Run this inside the Megatron-LM CUDA/TE test environment."
        ) from exc

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is unavailable in the selected environment.") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this CP reproduction requires two GPUs.")
    capability = torch.cuda.get_device_capability(local_rank)
    if capability not in {(9, 0), (10, 0), (10, 3)}:
        raise SystemExit(
            "The current CSA CP layout kernel supports sm_90a/sm_100a/sm_103a only; "
            f"local rank {local_rank} has compute capability {capability}."
        )

    plugin = _ForceUnfusedPlugin()
    pytest_args = [
        "-vv",
        "-s",
        "--tb=short",
        "--experimental",
        TARGET_TEST,
        "-k",
        _selection_expression(args.loss_mode),
    ]
    exit_code = int(pytest.main(pytest_args, plugins=[plugin]))

    if local_rank == 0:
        if not plugin.patched:
            print("[PR5960-CP2] INFRA ERROR: failed to force the unfused test path.", flush=True)
            return 4
        if plugin.failed_calls:
            print(
                "[PR5960-CP2] REPRODUCED: CP2 differs from the full-sequence CP1 "
                "reference on the unfused ratio-4 indexer path.",
                flush=True,
            )
        elif plugin.passed_calls:
            print(
                "[PR5960-CP2] PASS: CP2 matches CP1; the reported regression is not "
                "present at this checkout (or has been fixed).",
                flush=True,
            )
        else:
            print(
                f"[PR5960-CP2] INFRA ERROR: no test body ran "
                f"(setup skips={plugin.skipped_setups}). Check CUDA and Transformer Engine.",
                flush=True,
            )
            return 4
    return exit_code


def main() -> int:
    args = _parse_args()
    under_torchrun = "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ
    if args.worker or under_torchrun:
        return _run_worker(args)
    return _launch_two_workers(args)


if __name__ == "__main__":
    raise SystemExit(main())
