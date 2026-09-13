"""
Main entry point for the HackerRank Orchestrate
"Buy or Wait?" challenge.

Official execution:
    python3 code/main.py

Expected output:
    ./output.csv
"""

from __future__ import annotations

import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Make the repository root importable.
#
# This allows:
#     python3 code/main.py
#
# to work correctly while still allowing the rest of the project to use
# imports such as:
#     from code.orchestration import Orchestrator
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from code.orchestration import Orchestrator


def main() -> int:
    """
    Run the complete financial decision pipeline.
    """

    dataset_dir = PROJECT_ROOT / "dataset"
    output_path = PROJECT_ROOT / "output.csv"

    print("=" * 70)
    print("HackerRank Orchestrate - Buy or Wait?")
    print("=" * 70)
    print(f"Dataset : {dataset_dir}")
    print(f"Output  : {output_path}")
    print()

    try:
        orchestrator = Orchestrator(
            dataset_dir=dataset_dir,
            output_path=output_path,
        )

        orchestrator.run()

        print()
        print("=" * 70)
        print("Pipeline completed successfully.")
        print(f"Output written to: {output_path}")
        print("=" * 70)

        return 0

    except KeyboardInterrupt:
        print()
        print("Execution interrupted by user.")
        return 130

    except Exception as exc:
        print()
        print("=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)
        print(f"Error: {exc}")
        print()
        print("The full traceback is available for debugging.")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
