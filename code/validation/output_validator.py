"""
Validation and defensive normalization of the final output rows.

This module is the last schema-level guard before output.csv is written.

It validates:
- required columns
- request IDs
- numeric amount safety
- allowed affordability statuses
- allowed payment methods
- payment-plan formatting
- date formatting
- spending-change formatting
- explanation presence

It does not decide financial affordability itself. That responsibility belongs
to the forecaster and decision policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


class OutputValidationError(ValueError):
    """Raised when output rows cannot be safely normalized."""


@dataclass(frozen=True)
class ValidationIssue:
    """
    One validation problem discovered in an output row.
    """

    request_id: str
    field: str
    message: str


class OutputValidator:
    """
    Validate and normalize final prediction rows.

    The validator is intentionally conservative:
    invalid financial values are rejected rather than silently invented.
    """

    REQUIRED_COLUMNS = (
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    )

    ALLOWED_STATUSES = {
        "affordable_now",
        "affordable_with_plan",
        "affordable_later",
        "not_affordable",
    }

    ALLOWED_METHODS = {
        "full_payment",
        "partial_payment",
        "installments",
        "wait",
        "not_recommended",
    }

    PAYMENT_PLAN_NONE = "none"
    SPENDING_CHANGES_NONE = "none"

    DATE_PATTERN = re.compile(
        r"^\d{4}-\d{2}-\d{2}$"
    )

    PAYMENT_ENTRY_PATTERN = re.compile(
        r"^\d{4}-\d{2}-\d{2}:-?\d+(?:\.\d+)?$"
    )

    STOP_PATTERN = re.compile(
        r"^stop:[^:|]+$"
    )

    REDUCE_PATTERN = re.compile(
        r"^reduce_to:[^:|]+:-?\d+(?:\.\d+)?$"
    )

    MAX_EXPLANATION_LENGTH = 1000

    def validate_rows(
        self,
        rows: Sequence[Dict[str, Any]],
        requests: Optional[Iterable[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Validate and normalize a collection of output rows.

        Returns a new list. Input dictionaries are not mutated.
        """

        request_map = self._build_request_map(requests)

        normalized_rows: List[Dict[str, Any]] = []
        issues: List[ValidationIssue] = []

        for index, row in enumerate(rows):
            request_id = str(row.get("request_id", "")).strip()

            try:
                normalized = self.validate_row(
                    row=row,
                    request=request_map.get(request_id),
                )
                normalized_rows.append(normalized)
            except OutputValidationError as exc:
                issues.append(
                    ValidationIssue(
                        request_id=request_id or f"<row:{index}>",
                        field="row",
                        message=str(exc),
                    )
                )

        duplicate_ids = self._duplicate_request_ids(normalized_rows)

        for request_id in duplicate_ids:
            issues.append(
                ValidationIssue(
                    request_id=request_id,
                    field="request_id",
                    message="Duplicate request_id.",
                )
            )

        if request_map:
            expected_ids = set(request_map)
            actual_ids = {
                str(row["request_id"])
                for row in normalized_rows
            }

            for missing_id in sorted(expected_ids - actual_ids):
                issues.append(
                    ValidationIssue(
                        request_id=missing_id,
                        field="request_id",
                        message="Missing output row for request.",
                    )
                )

        if issues:
            raise OutputValidationError(
                self._format_issues(issues)
            )

        return normalized_rows

    def validate_row(
        self,
        row: Dict[str, Any],
        request: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Validate and normalize one output row.
        """

        if not isinstance(row, dict):
            raise OutputValidationError(
                "Output row must be a dictionary."
            )

        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in row
        ]

        if missing:
            raise OutputValidationError(
                "Missing required columns: "
                + ", ".join(missing)
            )

        request_id = self._validate_request_id(
            row["request_id"]
        )

        requested_amount = self._requested_amount(request)

        safe_amount = self._validate_amount_safe_to_pay(
            row["amount_safe_to_pay"],
            requested_amount,
        )

        status = self._validate_status(
            row["affordability_status"]
        )

        method = self._validate_method(
            row["recommended_payment_method"]
        )

        payment_plan = self._validate_payment_plan(
            row["payment_plan"],
        )

        earliest_date = self._validate_earliest_date(
            row["earliest_date_for_full_payment"]
        )

        spending_changes = self._validate_spending_changes(
            row["spending_changes_needed"]
        )

        explanation = self._validate_explanation(
            row["decision_explanation"]
        )

        normalized = {
            "request_id": request_id,
            "amount_safe_to_pay": safe_amount,
            "affordability_status": status,
            "recommended_payment_method": method,
            "payment_plan": payment_plan,
            "earliest_date_for_full_payment": earliest_date,
            "spending_changes_needed": spending_changes,
            "decision_explanation": explanation,
        }

        self._validate_cross_field_consistency(
            normalized,
            requested_amount=requested_amount,
        )

        return normalized

    # ------------------------------------------------------------------
    # Individual field validation
    # ------------------------------------------------------------------

    def _validate_request_id(self, value: Any) -> str:
        request_id = str(value).strip()

        if not request_id:
            raise OutputValidationError(
                "request_id cannot be empty."
            )

        if "|" in request_id or "\n" in request_id:
            raise OutputValidationError(
                "request_id contains invalid delimiter characters."
            )

        return request_id

    def _validate_amount_safe_to_pay(
        self,
        value: Any,
        requested_amount: Optional[float],
    ) -> float:
        try:
            amount = float(value)
        except (TypeError, ValueError):
            raise OutputValidationError(
                "amount_safe_to_pay must be numeric."
            )

        if not math.isfinite(amount):
            raise OutputValidationError(
                "amount_safe_to_pay must be finite."
            )

        if amount < -0.01:
            raise OutputValidationError(
                "amount_safe_to_pay cannot be negative."
            )

        if requested_amount is not None:
            if amount > requested_amount + 0.01:
                raise OutputValidationError(
                    "amount_safe_to_pay cannot exceed requested amount."
                )

        # Avoid negative zero and excessive floating-point noise.
        amount = max(0.0, amount)

        return self._round_money(amount)

    def _validate_status(self, value: Any) -> str:
        status = str(value).strip()

        if status not in self.ALLOWED_STATUSES:
            raise OutputValidationError(
                f"Invalid affordability_status: {status!r}"
            )

        return status

    def _validate_method(self, value: Any) -> str:
        method = str(value).strip()

        if method not in self.ALLOWED_METHODS:
            raise OutputValidationError(
                f"Invalid recommended_payment_method: {method!r}"
            )

        return method

    def _validate_payment_plan(self, value: Any) -> str:
        plan = str(value).strip()

        if not plan:
            raise OutputValidationError(
                "payment_plan cannot be empty."
            )

        if plan == self.PAYMENT_PLAN_NONE:
            return plan

        entries = plan.split("|")

        for entry in entries:
            entry = entry.strip()

            if not self.PAYMENT_ENTRY_PATTERN.fullmatch(entry):
                raise OutputValidationError(
                    "Invalid payment_plan entry: "
                    f"{entry!r}. Expected YYYY-MM-DD:amount."
                )

            payment_date_text, amount_text = entry.split(":", 1)

            parsed_date = self._parse_date(payment_date_text)

            if parsed_date is None:
                raise OutputValidationError(
                    f"Invalid payment date: {payment_date_text!r}"
                )

            try:
                amount = float(amount_text)
            except ValueError:
                raise OutputValidationError(
                    f"Invalid payment amount: {amount_text!r}"
                )

            if not math.isfinite(amount):
                raise OutputValidationError(
                    "Payment amount must be finite."
                )

            if amount <= 0:
                raise OutputValidationError(
                    "Payment amounts must be positive."
                )

        # Ensure chronological ordering.
        dates = [
            self._parse_date(entry.split(":", 1)[0])
            for entry in entries
        ]

        if dates != sorted(dates):
            raise OutputValidationError(
                "payment_plan dates must be chronological."
            )

        return "|".join(entries)

    def _validate_earliest_date(self, value: Any) -> str:
        text = str(value).strip()

        if text == "":
            return ""

        parsed = self._parse_date(text)

        if parsed is None:
            raise OutputValidationError(
                "earliest_date_for_full_payment must be "
                "YYYY-MM-DD or empty."
            )

        return text

    def _validate_spending_changes(self, value: Any) -> str:
        changes = str(value).strip()

        if not changes:
            raise OutputValidationError(
                "spending_changes_needed cannot be empty."
            )

        if changes == self.SPENDING_CHANGES_NONE:
            return changes

        entries = [
            item.strip()
            for item in changes.split("|")
            if item.strip()
        ]

        if not entries:
            raise OutputValidationError(
                "Invalid spending_changes_needed."
            )

        if len(entries) > 3:
            raise OutputValidationError(
                "At most three spending changes are allowed."
            )

        for entry in entries:
            if self.STOP_PATTERN.fullmatch(entry):
                continue

            if self.REDUCE_PATTERN.fullmatch(entry):
                amount_text = entry.rsplit(":", 1)[1]

                try:
                    amount = float(amount_text)
                except ValueError:
                    raise OutputValidationError(
                        f"Invalid reduction amount: {entry!r}"
                    )

                if not math.isfinite(amount) or amount < 0:
                    raise OutputValidationError(
                        f"Invalid reduction amount: {entry!r}"
                    )

                continue

            raise OutputValidationError(
                "Invalid spending change: "
                f"{entry!r}. Expected stop:<event_id> or "
                "reduce_to:<event_id>:<amount>."
            )

        return "|".join(entries)

    def _validate_explanation(self, value: Any) -> str:
        explanation = " ".join(
            str(value).strip().split()
        )

        if not explanation:
            raise OutputValidationError(
                "decision_explanation cannot be empty."
            )

        if len(explanation) > self.MAX_EXPLANATION_LENGTH:
            raise OutputValidationError(
                "decision_explanation is too long."
            )

        return explanation

    # ------------------------------------------------------------------
    # Cross-field checks
    # ------------------------------------------------------------------

    def _validate_cross_field_consistency(
        self,
        row: Dict[str, Any],
        requested_amount: Optional[float],
    ) -> None:
        status = row["affordability_status"]
        method = row["recommended_payment_method"]
        safe_amount = float(row["amount_safe_to_pay"])
        plan = row["payment_plan"]
        earliest = row["earliest_date_for_full_payment"]
        changes = row["spending_changes_needed"]

        # A non-recommended decision must not pretend there is a payment.
        if method == "not_recommended":
            if plan != self.PAYMENT_PLAN_NONE:
                raise OutputValidationError(
                    "not_recommended must have payment_plan=none."
                )

        # Waiting means no payment is made today.
        if method == "wait":
            if plan != self.PAYMENT_PLAN_NONE:
                raise OutputValidationError(
                    "wait must have payment_plan=none."
                )

        # If the request is currently fully affordable, the safe amount
        # should cover the requested amount.
        if (
            status == "affordable_now"
            and requested_amount is not None
            and safe_amount < requested_amount - 0.01
        ):
            raise OutputValidationError(
                "affordable_now requires amount_safe_to_pay "
                "to cover the requested amount."
            )

        # An affordable-now full payment should have a payment plan.
        if (
            status == "affordable_now"
            and method == "full_payment"
            and plan == self.PAYMENT_PLAN_NONE
        ):
            raise OutputValidationError(
                "full_payment cannot have payment_plan=none."
            )

        # If full payment is already safe, the earliest full-payment date
        # should normally be known.
        if (
            status == "affordable_now"
            and not earliest
        ):
            raise OutputValidationError(
                "affordable_now requires an earliest full-payment date."
            )

        # A later/plan decision should not claim an impossible zero payment
        # schedule unless the method is wait/not_recommended.
        if (
            method in {"full_payment", "partial_payment", "installments"}
            and plan == self.PAYMENT_PLAN_NONE
        ):
            raise OutputValidationError(
                f"{method} requires a payment plan."
            )

        # Spending changes are allowed for affordable_with_plan but should
        # never appear as an empty string.
        if changes == "":
            raise OutputValidationError(
                "spending_changes_needed cannot be empty."
            )

    # ------------------------------------------------------------------
    # Request helpers
    # ------------------------------------------------------------------

    def _build_request_map(
        self,
        requests: Optional[Iterable[Any]],
    ) -> Dict[str, Any]:
        if requests is None:
            return {}

        result = {}

        for request in requests:
            request_id = getattr(request, "request_id", None)

            if request_id is None and isinstance(request, dict):
                request_id = request.get("request_id")

            if request_id is None:
                continue

            result[str(request_id)] = request

        return result

    def _requested_amount(
        self,
        request: Optional[Any],
    ) -> Optional[float]:
        if request is None:
            return None

        value = getattr(
            request,
            "requested_amount",
            None,
        )

        if value is None and isinstance(request, dict):
            value = request.get("requested_amount")

        if value is None:
            return None

        try:
            amount = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(amount):
            return None

        return amount

    # ------------------------------------------------------------------
    # General utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(value: str) -> Optional[date]:
        try:
            if not OutputValidator.DATE_PATTERN.fullmatch(value):
                return None

            return date.fromisoformat(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _round_money(value: float) -> float:
        return round(float(value), 2)

    @staticmethod
    def _duplicate_request_ids(
        rows: Sequence[Dict[str, Any]],
    ) -> Set[str]:
        seen = set()
        duplicates = set()

        for row in rows:
            request_id = str(row["request_id"])

            if request_id in seen:
                duplicates.add(request_id)

            seen.add(request_id)

        return duplicates

    @staticmethod
    def _format_issues(
        issues: Sequence[ValidationIssue],
    ) -> str:
        lines = [
            "Output validation failed:"
        ]

        for issue in issues[:25]:
            lines.append(
                f"- request_id={issue.request_id}, "
                f"field={issue.field}: {issue.message}"
            )

        if len(issues) > 25:
            lines.append(
                f"- ... and {len(issues) - 25} more issue(s)"
            )

        return "\n".join(lines)