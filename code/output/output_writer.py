"""
Final output.csv writer.

Responsibilities:
- enforce the exact required column order
- write deterministic CSV output
- preserve numeric values consistently
- avoid partially-written output files
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


class OutputWriter:
    """
    Writes validated decision rows to the required output.csv format.
    """

    COLUMNS = (
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    )

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)

    def write(
        self,
        rows: Sequence[Dict[str, Any]],
    ) -> Path:
        """
        Write rows to output.csv.

        The file is written to a temporary sibling file first and then
        atomically replaced into place.
        """

        normalized_rows = [
            self._normalize_row(row)
            for row in rows
        ]

        self._ensure_parent_directory()

        temporary_path = self.output_path.with_suffix(
            self.output_path.suffix + ".tmp"
        )

        try:
            self._write_csv(
                temporary_path,
                normalized_rows,
            )

            os.replace(
                temporary_path,
                self.output_path,
            )

        finally:
            if temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

        return self.output_path

    def write_from_iterable(
        self,
        rows: Iterable[Dict[str, Any]],
    ) -> Path:
        """
        Convenience wrapper for generators/iterables.
        """

        return self.write(list(rows))

    def _write_csv(
        self,
        path: Path,
        rows: Sequence[Dict[str, Any]],
    ) -> None:
        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(self.COLUMNS),
                extrasaction="ignore",
                lineterminator="\n",
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        column: row.get(column, "")
                        for column in self.COLUMNS
                    }
                )

            handle.flush()
            os.fsync(handle.fileno())

    def _normalize_row(
        self,
        row: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalize representation without changing the decision itself.
        """

        normalized = dict(row)

        normalized["request_id"] = str(
            normalized.get("request_id", "")
        ).strip()

        normalized["amount_safe_to_pay"] = (
            self._format_amount(
                normalized.get("amount_safe_to_pay", 0)
            )
        )

        normalized["affordability_status"] = str(
            normalized.get("affordability_status", "")
        ).strip()

        normalized["recommended_payment_method"] = str(
            normalized.get(
                "recommended_payment_method",
                "",
            )
        ).strip()

        normalized["payment_plan"] = str(
            normalized.get("payment_plan", "none")
        ).strip()

        normalized["earliest_date_for_full_payment"] = str(
            normalized.get(
                "earliest_date_for_full_payment",
                "",
            )
        ).strip()

        normalized["spending_changes_needed"] = str(
            normalized.get(
                "spending_changes_needed",
                "none",
            )
        ).strip()

        normalized["decision_explanation"] = " ".join(
            str(
                normalized.get(
                    "decision_explanation",
                    "",
                )
            ).strip().split()
        )

        return {
            column: normalized.get(column, "")
            for column in self.COLUMNS
        }

    @staticmethod
    def _format_amount(value: Any) -> str:
        """
        Produce a stable two-decimal representation.

        Using a string here avoids CSV output such as:
            123.0
            123.00000000001

        while preserving the actual monetary value to cents.
        """

        try:
            amount = float(value)
        except (TypeError, ValueError):
            return str(value)

        if amount == 0:
            amount = 0.0

        return f"{amount:.2f}"

    def _ensure_parent_directory(self) -> None:
        parent = self.output_path.parent

        if parent == Path("."):
            return

        parent.mkdir(
            parents=True,
            exist_ok=True,
        )