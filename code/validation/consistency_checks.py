"""
Cross-field consistency checks for final financial decisions.

This module sits between the decision engine and output generation.

It checks relationships such as:

- affordable_now -> full payment is actually possible
- full_payment -> payment plan covers the request
- partial_payment -> exactly two payments
- installments -> a non-empty multi-payment schedule
- wait -> no payment is made immediately
- spending changes -> valid output syntax
- earliest full-payment date -> consistent with the payment schedule

This is intentionally separate from OutputValidator:
OutputValidator checks schema and field formatting, while this module checks
whether different fields make sense together.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class ConsistencyIssue:
    request_id: str
    rule: str
    message: str


class ConsistencyChecker:
    """
    Cross-field validation for a final decision.

    The checker is defensive and returns issues rather than changing a
    financial decision silently.
    """

    EPSILON = 0.02

    PAYMENT_ENTRY_PATTERN = re.compile(
        r"^(\d{4}-\d{2}-\d{2}):(\d+(?:\.\d+)?)$"
    )

    def check_row(
        self,
        row: Dict[str, Any],
        request: Optional[Any] = None,
        payment_options: Optional[Iterable[Any]] = None,
    ) -> List[ConsistencyIssue]:
        """
        Check one output row.
        """

        request_id = str(
            row.get("request_id", "")
        ).strip()

        issues: List[ConsistencyIssue] = []

        requested_amount = self._requested_amount(request)

        status = str(
            row.get("affordability_status", "")
        ).strip()

        method = str(
            row.get("recommended_payment_method", "")
        ).strip()

        safe_amount = self._float_or_none(
            row.get("amount_safe_to_pay")
        )

        payment_plan = str(
            row.get("payment_plan", "")
        ).strip()

        earliest_date = str(
            row.get("earliest_date_for_full_payment", "")
        ).strip()

        spending_changes = str(
            row.get("spending_changes_needed", "")
        ).strip()

        # --------------------------------------------------------------
        # Basic numeric relationship
        # --------------------------------------------------------------

        if safe_amount is None:
            issues.append(
                self._issue(
                    request_id,
                    "safe_amount_numeric",
                    "amount_safe_to_pay is not numeric.",
                )
            )
        else:
            if safe_amount < -self.EPSILON:
                issues.append(
                    self._issue(
                        request_id,
                        "safe_amount_nonnegative",
                        "amount_safe_to_pay is negative.",
                    )
                )

            if (
                requested_amount is not None
                and safe_amount > requested_amount + self.EPSILON
            ):
                issues.append(
                    self._issue(
                        request_id,
                        "safe_amount_cap",
                        "amount_safe_to_pay exceeds requested_amount.",
                    )
                )

        # --------------------------------------------------------------
        # Status/method relationships
        # --------------------------------------------------------------

        if status == "affordable_now":
            if method != "full_payment":
                issues.append(
                    self._issue(
                        request_id,
                        "affordable_now_method",
                        "affordable_now should use full_payment.",
                    )
                )

            if (
                requested_amount is not None
                and safe_amount is not None
                and safe_amount + self.EPSILON < requested_amount
            ):
                issues.append(
                    self._issue(
                        request_id,
                        "affordable_now_amount",
                        "affordable_now requires the full requested amount "
                        "to be safe immediately.",
                    )
                )

        if status == "affordable_with_plan":
            if method not in {
                "full_payment",
                "partial_payment",
                "installments",
            }:
                issues.append(
                    self._issue(
                        request_id,
                        "affordable_with_plan_method",
                        "affordable_with_plan requires a payment method "
                        "that actually makes payments.",
                    )
                )

        if status == "affordable_later":
            if method != "wait":
                issues.append(
                    self._issue(
                        request_id,
                        "affordable_later_method",
                        "affordable_later should use wait.",
                    )
                )

        if status == "not_affordable":
            if method != "not_recommended":
                issues.append(
                    self._issue(
                        request_id,
                        "not_affordable_method",
                        "not_affordable should use not_recommended.",
                    )
                )

        # --------------------------------------------------------------
        # Payment-plan relationship
        # --------------------------------------------------------------

        payments = self._parse_payment_plan(payment_plan)

        if method == "not_recommended":
            if payment_plan != "none":
                issues.append(
                    self._issue(
                        request_id,
                        "not_recommended_plan",
                        "not_recommended must have payment_plan=none.",
                    )
                )

        elif method == "wait":
            if payment_plan != "none":
                issues.append(
                    self._issue(
                        request_id,
                        "wait_plan",
                        "wait must have payment_plan=none.",
                    )
                )

        elif method in {
            "full_payment",
            "partial_payment",
            "installments",
        }:
            if not payments:
                issues.append(
                    self._issue(
                        request_id,
                        "payment_plan_required",
                        f"{method} requires a payment schedule.",
                    )
                )

        # --------------------------------------------------------------
        # Full payment
        # --------------------------------------------------------------

        if method == "full_payment" and payments:
            total = self._payment_total(payments)

            if (
                requested_amount is not None
                and abs(total - requested_amount) > self.EPSILON
            ):
                issues.append(
                    self._issue(
                        request_id,
                        "full_payment_total",
                        "full_payment schedule does not equal the "
                        "requested amount.",
                    )
                )

            if len(payments) != 1:
                issues.append(
                    self._issue(
                        request_id,
                        "full_payment_count",
                        "full_payment must contain exactly one payment.",
                    )
                )

        # --------------------------------------------------------------
        # Partial payment
        # --------------------------------------------------------------

        if method == "partial_payment":
            allows_partial = self._allows_partial_payment(request)

            if request is not None and not allows_partial:
                issues.append(
                    self._issue(
                        request_id,
                        "partial_payment_permission",
                        "Request does not allow partial payment.",
                    )
                )

            if len(payments) != 2:
                issues.append(
                    self._issue(
                        request_id,
                        "partial_payment_count",
                        "partial_payment must contain exactly two payments.",
                    )
                )
            elif requested_amount is not None:
                first_amount = payments[0][1]
                second_amount = payments[1][1]

                if abs(
                    first_amount + second_amount - requested_amount
                ) > self.EPSILON:
                    issues.append(
                        self._issue(
                            request_id,
                            "partial_payment_total",
                            "The two partial payments do not equal "
                            "the requested amount.",
                        )
                    )

                if first_amount <= self.EPSILON:
                    issues.append(
                        self._issue(
                            request_id,
                            "partial_payment_first_amount",
                            "The first partial payment must be positive.",
                        )
                    )

                if second_amount <= self.EPSILON:
                    issues.append(
                        self._issue(
                            request_id,
                            "partial_payment_second_amount",
                            "The second partial payment must be positive.",
                        )
                    )

            if payments:
                request_date = self._request_date(request)

                if request_date is not None:
                    if payments[0][0] != request_date:
                        issues.append(
                            self._issue(
                                request_id,
                                "partial_payment_start",
                                "The first partial payment must occur "
                                "on request_date.",
                            )
                        )

        # --------------------------------------------------------------
        # Installments
        # --------------------------------------------------------------

        if method == "installments":
            if len(payments) < 2:
                issues.append(
                    self._issue(
                        request_id,
                        "installment_count",
                        "installments requires at least two payments.",
                    )
                )

            if payments:
                total = self._payment_total(payments)

                if (
                    requested_amount is not None
                    and total + self.EPSILON < requested_amount
                ):
                    issues.append(
                        self._issue(
                            request_id,
                            "installment_total",
                            "Installment schedule does not cover "
                            "the requested amount.",
                        )
                    )

                self._check_supplied_installment_option(
                    issues=issues,
                    request_id=request_id,
                    payments=payments,
                    payment_options=payment_options,
                )

        # --------------------------------------------------------------
        # Earliest full-payment date
        # --------------------------------------------------------------

        if earliest_date:
            parsed_earliest = self._parse_date(
                earliest_date
            )

            if parsed_earliest is None:
                issues.append(
                    self._issue(
                        request_id,
                        "earliest_date_format",
                        "earliest_date_for_full_payment is not a valid date.",
                    )
                )
            elif payments:
                last_payment_date = payments[-1][0]

                # For an actual payment schedule, full payment must be
                # complete no later than the final payment date.
                if parsed_earliest > last_payment_date:
                    issues.append(
                        self._issue(
                            request_id,
                            "earliest_date_schedule",
                            "earliest_date_for_full_payment occurs after "
                            "the final payment in payment_plan.",
                        )
                    )

        # --------------------------------------------------------------
        # Spending changes
        # --------------------------------------------------------------

        self._check_spending_changes(
            issues=issues,
            request_id=request_id,
            spending_changes=spending_changes,
        )

        return issues

    def check_rows(
        self,
        rows: Sequence[Dict[str, Any]],
        requests: Optional[Iterable[Any]] = None,
        payment_options_by_request: Optional[Dict[str, Iterable[Any]]] = None,
    ) -> List[ConsistencyIssue]:
        """
        Check multiple rows and return every discovered issue.
        """

        request_map = self._build_request_map(requests)

        all_issues: List[ConsistencyIssue] = []

        for row in rows:
            request_id = str(
                row.get("request_id", "")
            ).strip()

            request = request_map.get(request_id)

            options = None

            if payment_options_by_request:
                options = payment_options_by_request.get(
                    request_id
                )

            all_issues.extend(
                self.check_row(
                    row=row,
                    request=request,
                    payment_options=options,
                )
            )

        return all_issues

    def assert_consistent(
        self,
        row: Dict[str, Any],
        request: Optional[Any] = None,
        payment_options: Optional[Iterable[Any]] = None,
    ) -> None:
        """
        Raise ValueError if a row violates any consistency rule.
        """

        issues = self.check_row(
            row=row,
            request=request,
            payment_options=payment_options,
        )

        if not issues:
            return

        lines = [
            "Decision consistency check failed:"
        ]

        for issue in issues:
            lines.append(
                f"- [{issue.rule}] {issue.message}"
            )

        raise ValueError("\n".join(lines))

    # ------------------------------------------------------------------
    # Supplied installment validation
    # ------------------------------------------------------------------

    def _check_supplied_installment_option(
        self,
        issues: List[ConsistencyIssue],
        request_id: str,
        payments: List[tuple[date, float]],
        payment_options: Optional[Iterable[Any]],
    ) -> None:
        """
        If supplied payment options are available, require the emitted
        installment schedule to exactly match one of them.

        This prevents the agent from inventing an installment schedule.
        """

        if payment_options is None:
            return

        options = list(payment_options)

        if not options:
            issues.append(
                self._issue(
                    request_id,
                    "installment_option_missing",
                    "installments were selected but no supplied "
                    "payment option exists.",
                )
            )
            return

        matching_option = False

        emitted_dates = [
            payment_date
            for payment_date, _ in payments
        ]

        emitted_amounts = [
            amount
            for _, amount in payments
        ]

        for option in options:
            option_method = self._option_value(
                option,
                "payment_method",
            )

            if option_method != "installments":
                continue

            option_schedule = self._option_schedule(
                option
            )

            if not option_schedule:
                continue

            option_dates = [
                payment_date
                for payment_date, _ in option_schedule
            ]

            option_amounts = [
                amount
                for _, amount in option_schedule
            ]

            if len(option_schedule) != len(payments):
                continue

            if emitted_dates != option_dates:
                continue

            if all(
                abs(a - b) <= self.EPSILON
                for a, b in zip(
                    emitted_amounts,
                    option_amounts,
                )
            ):
                matching_option = True
                break

        if not matching_option:
            issues.append(
                self._issue(
                    request_id,
                    "installment_exact_match",
                    "Installment schedule does not exactly match "
                    "any supplied installment option.",
                )
            )

    def _option_schedule(
        self,
        option: Any,
    ) -> List[tuple[date, float]]:
        amount = self._float_or_none(
            self._option_value(
                option,
                "payment_amount",
            )
        )

        count_value = self._option_value(
            option,
            "number_of_payments",
        )

        first_date = self._parse_date(
            str(
                self._option_value(
                    option,
                    "first_payment_date",
                ) or ""
            )
        )

        frequency = self._float_or_none(
            self._option_value(
                option,
                "payment_frequency_days",
            )
        )

        if (
            amount is None
            or count_value is None
            or first_date is None
            or frequency is None
        ):
            return []

        try:
            count = int(count_value)
        except (TypeError, ValueError):
            return []

        if count <= 0 or frequency < 0:
            return []

        from datetime import timedelta

        return [
            (
                first_date + timedelta(
                    days=int(frequency * index)
                ),
                float(amount),
            )
            for index in range(count)
        ]

    # ------------------------------------------------------------------
    # Spending-change checks
    # ------------------------------------------------------------------

    def _check_spending_changes(
        self,
        issues: List[ConsistencyIssue],
        request_id: str,
        spending_changes: str,
    ) -> None:
        if spending_changes == "none":
            return

        if not spending_changes:
            issues.append(
                self._issue(
                    request_id,
                    "spending_changes_present",
                    "spending_changes_needed is empty.",
                )
            )
            return

        entries = [
            entry.strip()
            for entry in spending_changes.split("|")
            if entry.strip()
        ]

        if len(entries) > 3:
            issues.append(
                self._issue(
                    request_id,
                    "spending_change_limit",
                    "No more than three spending changes are allowed.",
                )
            )

        for entry in entries:
            if entry.startswith("stop:"):
                if entry.count(":") != 1:
                    issues.append(
                        self._issue(
                            request_id,
                            "spending_change_format",
                            f"Invalid stop change: {entry}",
                        )
                    )
                continue

            if entry.startswith("reduce_to:"):
                pieces = entry.split(":")

                if len(pieces) != 3:
                    issues.append(
                        self._issue(
                            request_id,
                            "spending_change_format",
                            f"Invalid reduce change: {entry}",
                        )
                    )
                    continue

                amount = self._float_or_none(
                    pieces[2]
                )

                if amount is None or amount < 0:
                    issues.append(
                        self._issue(
                            request_id,
                            "spending_change_amount",
                            f"Invalid reduction amount: {entry}",
                        )
                    )

                continue

            issues.append(
                self._issue(
                    request_id,
                    "spending_change_format",
                    f"Unknown spending change: {entry}",
                )
            )

    # ------------------------------------------------------------------
    # Request helpers
    # ------------------------------------------------------------------

    def _requested_amount(
        self,
        request: Optional[Any],
    ) -> Optional[float]:
        return self._request_float(
            request,
            "requested_amount",
        )

    def _request_date(
        self,
        request: Optional[Any],
    ) -> Optional[date]:
        value = self._request_value(
            request,
            "request_date",
        )

        return self._parse_date(
            str(value)
            if value is not None
            else ""
        )

    def _allows_partial_payment(
        self,
        request: Optional[Any],
    ) -> bool:
        value = self._request_value(
            request,
            "allows_partial_payment",
        )

        if isinstance(value, bool):
            return value

        if value is None:
            return False

        return str(value).strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    # ------------------------------------------------------------------
    # Generic object/dict helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _request_value(
        request: Optional[Any],
        field: str,
    ):
        if request is None:
            return None

        value = getattr(
            request,
            field,
            None,
        )

        if value is None and isinstance(request, dict):
            value = request.get(field)

        return value

    def _request_float(
        self,
        request: Optional[Any],
        field: str,
    ) -> Optional[float]:
        return self._float_or_none(
            self._request_value(
                request,
                field,
            )
        )

    @staticmethod
    def _option_value(
        option: Any,
        field: str,
    ):
        if isinstance(option, dict):
            return option.get(field)

        return getattr(
            option,
            field,
            None,
        )

    # ------------------------------------------------------------------
    # Payment-plan parsing
    # ------------------------------------------------------------------

    def _parse_payment_plan(
        self,
        payment_plan: str,
    ) -> List[tuple[date, float]]:
        if payment_plan == "none" or not payment_plan:
            return []

        result = []

        for entry in payment_plan.split("|"):
            entry = entry.strip()

            match = self.PAYMENT_ENTRY_PATTERN.fullmatch(
                entry
            )

            if not match:
                return []

            payment_date = self._parse_date(
                match.group(1)
            )

            amount = self._float_or_none(
                match.group(2)
            )

            if payment_date is None or amount is None:
                return []

            if amount <= 0:
                return []

            result.append(
                (
                    payment_date,
                    amount,
                )
            )

        return result

    @staticmethod
    def _payment_total(
        payments: Sequence[tuple[date, float]],
    ) -> float:
        return round(
            sum(amount for _, amount in payments),
            2,
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _float_or_none(
        value: Any,
    ) -> Optional[float]:
        try:
            if value is None:
                return None

            result = float(value)

            if not math.isfinite(result):
                return None

            return result
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_date(
        value: str,
    ) -> Optional[date]:
        try:
            return date.fromisoformat(
                str(value).strip()
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_request_map(
        requests: Optional[Iterable[Any]],
    ) -> Dict[str, Any]:
        if requests is None:
            return {}

        result = {}

        for request in requests:
            if isinstance(request, dict):
                request_id = request.get("request_id")
            else:
                request_id = getattr(
                    request,
                    "request_id",
                    None,
                )

            if request_id is not None:
                result[str(request_id)] = request

        return result

    @staticmethod
    def _issue(
        request_id: str,
        rule: str,
        message: str,
    ) -> ConsistencyIssue:
        return ConsistencyIssue(
            request_id=request_id,
            rule=rule,
            message=message,
        )