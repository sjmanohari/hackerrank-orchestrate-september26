"""
HackerRank Orchestrate — Buy or Wait?

Official entry point.

Run from repository root:
    python3 code/main.py

The program:
1. Loads all datasets from ./dataset
2. Runs the financial decision pipeline
3. Validates every generated row
4. Writes ./output.csv
5. Writes ./evaluation/usage_report.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent

# Ensure `code.*` imports resolve correctly when this file is executed as:
#     python3 code/main.py
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Also support execution from inside code/:
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from code.orchestration import Orchestrator


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _load_dotenv_if_available() -> None:
    """
    Load .env when python-dotenv is installed.

    The solution does not require dotenv; environment variables already
    present in the execution environment continue to work normally.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except Exception:
        # dotenv is optional.
        pass


def _ensure_directories() -> None:
    """
    Create directories required for generated artifacts.
    """
    (REPO_ROOT / "evaluation").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _load_dotenv_if_available()
    _ensure_directories()

    dataset_root = REPO_ROOT / "dataset"
    output_path = REPO_ROOT / "output.csv"
    usage_report_path = REPO_ROOT / "evaluation" / "usage_report.md"

    if not dataset_root.exists():
        print(
            f"ERROR: dataset directory not found: {dataset_root}",
            file=sys.stderr,
        )
        return 1

    requests_path = dataset_root / "requests.csv"

    if not requests_path.exists():
        print(
            f"ERROR: required input file not found: {requests_path}",
            file=sys.stderr,
        )
        return 1

    print("=" * 72)
    print("HackerRank Orchestrate — Buy or Wait?")
    print("=" * 72)
    print(f"Repository : {REPO_ROOT}")
    print(f"Dataset    : {dataset_root}")
    print(f"Output     : {output_path}")
    print(f"Usage      : {usage_report_path}")
    print("-" * 72)

    try:
        orchestrator = Orchestrator(
            dataset_root=dataset_root,
            output_path=output_path,
            usage_report_path=usage_report_path,
        )

        result = orchestrator.run()

        # `run()` may return a dictionary, a list, or None depending on the
        # orchestration implementation. Keep the entry point tolerant.
        if isinstance(result, dict):
            processed = result.get("processed_requests")
            output = result.get("output_path")

            if processed is not None:
                print(f"Processed  : {processed} requests")

            if output:
                print(f"Output     : {output}")

        elif isinstance(result, list):
            print(f"Processed  : {len(result)} requests")

        print("-" * 72)

        if output_path.exists():
            print(f"SUCCESS: generated {output_path}")
        else:
            print(
                "ERROR: pipeline completed without creating output.csv",
                file=sys.stderr,
            )
            return 1

        if usage_report_path.exists():
            print(f"Usage report: {usage_report_path}")
        else:
            print(
                "WARNING: usage report was not generated.",
                file=sys.stderr,
            )

        print("=" * 72)

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    except Exception as exc:
        print("=" * 72, file=sys.stderr)
        print("PIPELINE FAILED", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)

        # Optional traceback for debugging. Disabled by default so the normal
        # submission output stays concise.
        if os.getenv("DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            import traceback

            traceback.print_exc()

        return 1


if __name__ == "__main__":
    raise SystemExit(main())