from __future__ import annotations

from datetime import date
from typing import Iterable, Optional

from .decision_policy import Decision
from ..normalization.models import (
    FinancialEvent,
    FinancialRequest,
    UserContext,
)


class ExplanationBuilder:
    """
    Builds concise, deterministic explanations for final decisions.

    The explanation is derived only from:
      * the request
      * the user's financial context
      * the computed decision
      * relevant financial events

    No new financial facts are invented here.
    """

    def __init__(self) -> None:
        pass

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def build(
        self,
        request: FinancialRequest,
        context: UserContext,
        decision: Decision,
        events: Optional[Iterable[FinancialEvent]] = None,
    ) -> str:
        """
        Build the final decision explanation.

        The output is intentionally short because it will be placed
        directly into output.csv.
        """

        events = list(events or [])

        requested_amount = self._money(
            getattr(
                request,
                "requested_amount",
                0,
            )
        )

        safe_amount = self._money(
            decision.amount_safe_to_pay
        )

        status = self._normalize(
            decision.affordability_status
        )

        method = self._normalize(
            decision.recommended_payment_method
        )

        changes = self._parse_changes(
            decision.spending_changes_needed
        )

        # --------------------------------------------------------------
        # AFFORDABLE NOW
        # --------------------------------------------------------------

        if status == "affordable_now":
            if method == "full_payment":
                return self._full_payment_now(
                    requested_amount=requested_amount,
                    events=events,
                )

            return self._generic_now(
                requested_amount=requested_amount,
                method=method,
            )

        # --------------------------------------------------------------
        # AFFORDABLE WITH PLAN
        # --------------------------------------------------------------

        if status == "affordable_with_plan":
            if method == "partial_payment":
                return self._partial_payment(
                    requested_amount=requested_amount,
                    safe_amount=safe_amount,
                    decision=decision,
                )

            if method == "installments":
                return self._installments(
                    requested_amount=requested_amount,
                    decision=decision,
                )

            if changes:
                return self._with_spending_changes(
                    requested_amount=requested_amount,
                    method=method,
                    changes=changes,
                    decision=decision,
                )

            return self._generic_plan(
                requested_amount=requested_amount,
                method=method,
                decision=decision,
            )

        # --------------------------------------------------------------
        # AFFORDABLE LATER
        # --------------------------------------------------------------

        if status == "affordable_later":
            return self._affordable_later(
                requested_amount=requested_amount,
                decision=decision,
            )

        # --------------------------------------------------------------
        # NOT AFFORDABLE
        # --------------------------------------------------------------

        if status == "not_affordable":
            return self._not_affordable(
                requested_amount=requested_amount,
                decision=decision,
            )

        # --------------------------------------------------------------
        # SAFE FALLBACK
        # --------------------------------------------------------------

        return self._fallback(
            requested_amount=requested_amount,
            decision=decision,
        )

    # ==================================================================
    # AFFORDABLE NOW
    # ==================================================================

    def _full_payment_now(
        self,
        requested_amount: float,
        events: list[FinancialEvent],
    ) -> str:
        return (
            f"Safe to pay {self._amount(requested_amount)} "
            f"in full on the request date while maintaining "
            f"the required balance."
        )

    def _generic_now(
        self,
        requested_amount: float,
        method: str,
    ) -> str:
        method_text = self._method_text(
            method
        )

        return (
            f"The requested "
            f"{self._amount(requested_amount)} "
            f"is affordable now using {method_text}."
        )

    # ==================================================================
    # PARTIAL PAYMENT
    # ==================================================================

    def _partial_payment(
        self,
        requested_amount: float,
        safe_amount: float,
        decision: Decision,
    ) -> str:
        remaining = max(
            0.0,
            requested_amount - safe_amount,
        )

        completion_date = (
            decision.earliest_date_for_full_payment
        )

        if completion_date:
            return (
                f"Pay {self._amount(safe_amount)} now and "
                f"the remaining {self._amount(remaining)} "
                f"by {completion_date}; this stays within "
                f"the projected safe balance."
            )

        return (
            f"Pay {self._amount(safe_amount)} now and "
            f"the remaining {self._amount(remaining)} "
            f"later when funds are sufficient."
        )

    # ==================================================================
    # INSTALLMENTS
    # ==================================================================

    def _installments(
        self,
        requested_amount: float,
        decision: Decision,
    ) -> str:
        plan = decision.payment_plan

        if plan and plan != "none":
            payment_count = len(
                [
                    item
                    for item in plan.split("|")
                    if item.strip()
                ]
            )

            completion_date = (
                decision.earliest_date_for_full_payment
            )

            if completion_date:
                return (
                    f"Use the supplied installment plan for "
                    f"{self._amount(requested_amount)} "
                    f"across {payment_count} payments, "
                    f"completing by {completion_date}."
                )

            return (
                f"Use the supplied installment plan for "
                f"{self._amount(requested_amount)} "
                f"across {payment_count} payments."
            )

        return (
            f"Use the supplied installment option because "
            f"it keeps the purchase within the projected "
            f"safe balance."
        )

    # ==================================================================
    # SPENDING CHANGES
    # ==================================================================

    def _with_spending_changes(
        self,
        requested_amount: float,
        method: str,
        changes: list[str],
        decision: Decision,
    ) -> str:
        action_text = self._describe_changes(
            changes
        )

        method_text = self._method_text(
            method
        )

        if decision.earliest_date_for_full_payment:
            return (
                f"Use {method_text} after {action_text}; "
                f"the purchase can then be completed by "
                f"{decision.earliest_date_for_full_payment}."
            )

        return (
            f"Use {method_text} after {action_text}; "
            f"the changes create enough projected room "
            f"for the purchase."
        )

    def _describe_changes(
        self,
        changes: list[str],
    ) -> str:
        descriptions: list[str] = []

        for change in changes[:3]:
            parts = change.split(":")

            if not parts:
                continue

            action = parts[0].strip().lower()

            if action == "stop" and len(parts) >= 2:
                descriptions.append(
                    f"stopping {parts[1]}"
                )

            elif (
                action == "reduce_to"
                and len(parts) >= 3
            ):
                descriptions.append(
                    f"reducing {parts[1]} to "
                    f"{parts[2]}"
                )

        if not descriptions:
            return "the permitted spending changes"

        if len(descriptions) == 1:
            return descriptions[0]

        if len(descriptions) == 2:
            return (
                f"{descriptions[0]} and "
                f"{descriptions[1]}"
            )

        return (
            f"{descriptions[0]}, "
            f"{descriptions[1]}, and "
            f"{descriptions[2]}"
        )

    # ==================================================================
    # AFFORDABLE LATER
    # ==================================================================

    def _affordable_later(
        self,
        requested_amount: float,
        decision: Decision,
    ) -> str:
        date_text = (
            decision.earliest_date_for_full_payment
        )

        if date_text:
            return (
                f"Wait until {date_text}; "
                f"the requested {self._amount(requested_amount)} "
                f"is not safely payable today but is projected "
                f"to become affordable then."
            )

        return (
            f"Wait until sufficient funds are available "
            f"for the requested "
            f"{self._amount(requested_amount)}."
        )

    # ==================================================================
    # NOT AFFORDABLE
    # ==================================================================

    def _not_affordable(
        self,
        requested_amount: float,
        decision: Decision,
    ) -> str:
        if decision.spending_changes_needed not in {
            "",
            "none",
        }:
            return (
                f"The requested {self._amount(requested_amount)} "
                f"cannot be safely covered under the available "
                f"constraints even with the permitted spending "
                f"changes."
            )

        return (
            f"The requested {self._amount(requested_amount)} "
            f"cannot be safely covered while maintaining the "
            f"required financial buffer."
        )

    # ==================================================================
    # GENERIC PLAN
    # ==================================================================

    def _generic_plan(
        self,
        requested_amount: float,
        method: str,
        decision: Decision,
    ) -> str:
        method_text = self._method_text(
            method
        )

        if decision.earliest_date_for_full_payment:
            return (
                f"Use {method_text} for the "
                f"{self._amount(requested_amount)} "
                f"purchase, completing by "
                f"{decision.earliest_date_for_full_payment}."
            )

        return (
            f"Use {method_text} because the purchase remains "
            f"within the projected safe balance."
        )

    # ==================================================================
    # FALLBACK
    # ==================================================================

    def _fallback(
        self,
        requested_amount: float,
        decision: Decision,
    ) -> str:
        if decision.reason:
            return self._clean(
                decision.reason
            )

        return (
            f"Decision based on the projected safe balance "
            f"for the requested "
            f"{self._amount(requested_amount)}."
        )

    # ==================================================================
    # METHOD WORDING
    # ==================================================================

    def _method_text(
        self,
        method: str,
    ) -> str:
        mapping = {
            "full_payment": "full payment",
            "partial_payment": "partial payment",
            "installments": "the supplied installment plan",
            "wait": "waiting",
            "not_recommended": "no payment method",
        }

        return mapping.get(
            method,
            method.replace(
                "_",
                " ",
            ),
        )

    # ==================================================================
    # SPENDING CHANGE PARSING
    # ==================================================================

    def _parse_changes(
        self,
        value: str,
    ) -> list[str]:
        if value is None:
            return []

        value = str(value).strip()

        if not value or value.lower() == "none":
            return []

        return [
            item.strip()
            for item in value.split("|")
            if item.strip()
        ]

    # ==================================================================
    # NORMALIZATION
    # ==================================================================

    @staticmethod
    def _normalize(
        value,
    ) -> str:
        if value is None:
            return ""

        return (
            str(value)
            .strip()
            .lower()
            .replace(
                "-",
                "_",
            )
            .replace(
                " ",
                "_",
            )
        )

    @staticmethod
    def _money(
        value,
    ) -> float:
        if value is None:
            return 0.0

        try:
            return round(
                float(value),
                2,
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    @classmethod
    def _amount(
        cls,
        value,
    ) -> str:
        value = cls._money(value)

        if abs(value) < 0.005:
            return "0"

        if value.is_integer():
            return f"{int(value):,}"

        return (
            f"{value:,.2f}"
            .rstrip("0")
            .rstrip(".")
        )

    @staticmethod
    def _date_only(
        value,
    ) -> Optional[date]:
        if value is None:
            return None

        if isinstance(
            value,
            date,
        ):
            return value

        text = str(value).strip()

        if not text:
            return None

        try:
            from datetime import datetime

            return datetime.fromisoformat(
                text.replace(
                    "Z",
                    "+00:00",
                )
            ).date()
        except ValueError:
            pass

        try:
            return date.fromisoformat(
                text[:10]
            )
        except ValueError:
            return None

    @staticmethod
    def _clean(
        text: str,
    ) -> str:
        """
        Prevent accidental CSV-breaking whitespace/newlines.
        """

        return (
            str(text)
            .replace(
                "\n",
                " ",
            )
            .replace(
                "\r",
                " ",
            )
            .strip()
        )