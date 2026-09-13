from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from ..normalization.models import FinancialEvent, UserContext


@dataclass(frozen=True)
class SpendingChange:
    """
    A permitted change to a future flexible expense.

    action:
        stop
        reduce_to

    output format:
        stop:<event_id>
        reduce_to:<event_id>:<new_amount>
    """

    event_id: str
    action: str
    new_amount: Optional[float] = None

    def output(self) -> str:
        if self.action == "stop":
            return f"stop:{self.event_id}"

        if self.action == "reduce_to":
            amount = self.new_amount if self.new_amount is not None else 0.0
            return f"reduce_to:{self.event_id}:{_format_amount(amount)}"

        raise ValueError(f"Unsupported spending-change action: {self.action}")


@dataclass(frozen=True)
class AdjustmentScenario:
    """
    A complete set of spending changes considered by the decision engine.
    """

    changes: tuple[SpendingChange, ...]
    total_reduction: float

    def outputs(self) -> list[str]:
        return [change.output() for change in self.changes]

    def output(self) -> str:
        if not self.changes:
            return "none"

        return "|".join(self.outputs())


class SpendingAdjustmentEngine:
    """
    Generates safe, user-authorized spending-adjustment scenarios.

    Important rules:
      * Never modify protected categories.
      * Only reduce categories the user explicitly permits.
      * Only stop categories the user explicitly permits.
      * Never reduce below the event's minimum_allowed_amount.
      * Never create a change for a fixed/protected expense.
      * Do not modify already-settled historical expenses.
      * Do not combine stop and reduce for the same event.
      * Keep scenarios deterministic.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def candidate_changes(
        self,
        events: Iterable[FinancialEvent],
        context: UserContext,
        as_of_date=None,
    ) -> list[SpendingChange]:
        """
        Return all individually valid spending changes.

        The result is deterministic and sorted by:
          1. largest immediate reduction
          2. stop before reduce
          3. event date
          4. event_id
        """

        candidates: list[tuple[SpendingChange, float, FinancialEvent]] = []

        protected = self._normalized_categories(
            getattr(context, "protected_categories", [])
        )

        reduce_categories = self._normalized_categories(
            self._context_reduce_categories(context)
        )

        stop_categories = self._normalized_categories(
            self._context_stop_categories(context)
        )

        for event in events:
            if not self._eligible_event(event, as_of_date):
                continue

            category = self._normalize(event.category)

            # Protected categories always win.
            if category in protected:
                continue

            amount = self._safe_float(event.amount)
            if amount <= 0:
                continue

            flexibility = self._normalize(
                getattr(event, "flexibility", "")
            )

            # ----------------------------------------------------------
            # STOP
            # ----------------------------------------------------------
            if category in stop_categories and self._can_stop(event):
                change = SpendingChange(
                    event_id=event.event_id,
                    action="stop",
                    new_amount=0.0,
                )

                candidates.append(
                    (
                        change,
                        amount,
                        event,
                    )
                )

                # Stop and reduce are mutually exclusive for the same
                # event. Do not create a second independent candidate
                # when the event is stop-eligible.
                continue

            # ----------------------------------------------------------
            # REDUCE
            # ----------------------------------------------------------
            if category in reduce_categories and self._can_reduce(
                event,
                flexibility,
            ):
                minimum = self._minimum_allowed_amount(event)

                if minimum < amount:
                    change = SpendingChange(
                        event_id=event.event_id,
                        action="reduce_to",
                        new_amount=minimum,
                    )

                    candidates.append(
                        (
                            change,
                            amount - minimum,
                            event,
                        )
                    )

        candidates.sort(
            key=lambda item: (
                -item[1],
                0 if item[0].action == "stop" else 1,
                self._date_key(item[2].event_date),
                item[2].event_id,
            )
        )

        return [item[0] for item in candidates]

    def generate_scenarios(
        self,
        events: Iterable[FinancialEvent],
        context: UserContext,
        as_of_date=None,
        max_changes: int = 3,
    ) -> list[AdjustmentScenario]:
        """
        Generate deterministic combinations of permitted changes.

        The challenge allows at most three spending changes in the final
        output. We therefore generate combinations of size 1..max_changes.

        The empty scenario is always included first.
        """

        max_changes = max(0, min(int(max_changes), 3))

        candidates = self.candidate_changes(
            events=events,
            context=context,
            as_of_date=as_of_date,
        )

        scenarios: list[AdjustmentScenario] = [
            AdjustmentScenario(
                changes=tuple(),
                total_reduction=0.0,
            )
        ]

        if not candidates or max_changes == 0:
            return scenarios

        # Avoid importing itertools globally just for this small operation.
        from itertools import combinations

        # Do not allow more changes than there are candidates.
        upper = min(max_changes, len(candidates))

        for size in range(1, upper + 1):
            for combo in combinations(candidates, size):
                if not self._compatible_changes(combo):
                    continue

                reduction = self._scenario_reduction(
                    combo,
                    events,
                )

                if reduction <= 0:
                    continue

                scenarios.append(
                    AdjustmentScenario(
                        changes=tuple(combo),
                        total_reduction=reduction,
                    )
                )

        scenarios.sort(
            key=lambda scenario: (
                len(scenario.changes),
                -scenario.total_reduction,
                scenario.output(),
            )
        )

        return scenarios

    def apply_changes(
        self,
        events: Iterable[FinancialEvent],
        changes: Iterable[SpendingChange],
    ) -> list[FinancialEvent]:
        """
        Apply spending changes to a copy of the event list.

        The original event objects are never mutated.

        This method changes only the amount of the matching future event.
        It does not change historical records, statuses, currencies,
        descriptions, or dates.
        """

        from dataclasses import replace

        changes_by_event = {
            change.event_id: change
            for change in changes
        }

        adjusted: list[FinancialEvent] = []

        for event in events:
            change = changes_by_event.get(event.event_id)

            if change is None:
                adjusted.append(event)
                continue

            amount = self._safe_float(event.amount)

            if change.action == "stop":
                adjusted.append(
                    replace(
                        event,
                        amount=0.0,
                    )
                )
                continue

            if change.action == "reduce_to":
                new_amount = (
                    self._safe_float(change.new_amount)
                )

                minimum = self._minimum_allowed_amount(event)

                # Defensive protection against malformed changes.
                new_amount = max(minimum, min(amount, new_amount))

                adjusted.append(
                    replace(
                        event,
                        amount=new_amount,
                    )
                )
                continue

            # Unknown changes are ignored rather than silently mutating
            # the event.
            adjusted.append(event)

        return adjusted

    def validate_changes(
        self,
        changes: Iterable[SpendingChange],
        events: Iterable[FinancialEvent],
        context: UserContext,
    ) -> tuple[bool, str]:
        """
        Validate a proposed spending-change set.

        Returns:
            (True, "") when valid
            (False, reason) otherwise
        """

        changes = list(changes)
        events_by_id = {
            event.event_id: event
            for event in events
        }

        if len(changes) > 3:
            return False, "At most three spending changes are allowed."

        seen: set[str] = set()

        protected = self._normalized_categories(
            getattr(context, "protected_categories", [])
        )

        reduce_categories = self._normalized_categories(
            self._context_reduce_categories(context)
        )

        stop_categories = self._normalized_categories(
            self._context_stop_categories(context)
        )

        for change in changes:
            if change.event_id in seen:
                return False, (
                    f"Duplicate spending change for {change.event_id}."
                )

            seen.add(change.event_id)

            event = events_by_id.get(change.event_id)

            if event is None:
                return False, (
                    f"Unknown spending event: {change.event_id}."
                )

            category = self._normalize(event.category)

            if category in protected:
                return False, (
                    f"Protected category cannot be changed: "
                    f"{event.event_id}."
                )

            if change.action == "stop":
                if category not in stop_categories:
                    return False, (
                        f"User did not authorize stopping category "
                        f"{event.category}."
                    )

                if not self._can_stop(event):
                    return False, (
                        f"Event cannot be stopped: {event.event_id}."
                    )

                continue

            if change.action == "reduce_to":
                if category not in reduce_categories:
                    return False, (
                        f"User did not authorize reducing category "
                        f"{event.category}."
                    )

                if not self._can_reduce(
                    event,
                    self._normalize(
                        getattr(event, "flexibility", "")
                    ),
                ):
                    return False, (
                        f"Event cannot be reduced: {event.event_id}."
                    )

                amount = self._safe_float(event.amount)
                new_amount = self._safe_float(change.new_amount)

                minimum = self._minimum_allowed_amount(event)

                if new_amount < minimum - 1e-9:
                    return False, (
                        f"Reduction below minimum allowed amount for "
                        f"{event.event_id}."
                    )

                if new_amount >= amount - 1e-9:
                    return False, (
                        f"Reduction does not reduce event amount: "
                        f"{event.event_id}."
                    )

                continue

            return False, (
                f"Unsupported spending change action: {change.action}."
            )

        return True, ""

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------

    def _eligible_event(
        self,
        event: FinancialEvent,
        as_of_date=None,
    ) -> bool:
        """
        Determine whether an event can be modified as a future spending
        event.
        """

        if not event.event_id:
            return False

        amount = self._safe_float(event.amount)
        if amount <= 0:
            return False

        direction = self._normalize(
            getattr(event, "direction", "")
        )

        # Only expenses/debits are spending changes.
        if direction not in {
            "debit",
            "expense",
            "outflow",
            "payment",
        }:
            return False

        status = self._normalize(
            getattr(event, "status", "")
        )

        if status in {
            "cancelled",
            "canceled",
            "reversed",
            "failed",
        }:
            return False

        # Already-settled historical expenses should not be altered.
        if status == "settled" and as_of_date is not None:
            event_date = self._date_only(
                getattr(event, "settlement_date", None)
                or getattr(event, "event_date", None)
            )

            reference_date = self._date_only(as_of_date)

            if (
                event_date is not None
                and reference_date is not None
                and event_date < reference_date
            ):
                return False

        return True

    def _can_stop(self, event: FinancialEvent) -> bool:
        flexibility = self._normalize(
            getattr(event, "flexibility", "")
        )

        return flexibility in {
            "stoppable",
            "stop",
            "can_stop",
            "optional",
            "flexible",
            "reducible_and_stoppable",
        }

    def _can_reduce(
        self,
        event: FinancialEvent,
        flexibility: str,
    ) -> bool:
        return flexibility in {
            "reducible",
            "reduce",
            "can_reduce",
            "optional",
            "flexible",
            "reducible_and_stoppable",
        }

    # ------------------------------------------------------------------
    # Scenario compatibility
    # ------------------------------------------------------------------

    def _compatible_changes(
        self,
        changes: Iterable[SpendingChange],
    ) -> bool:
        """
        Prevent contradictory operations.

        In particular:
            stop:event_1
            reduce_to:event_1:100

        must never occur in the same scenario.
        """

        seen: dict[str, str] = {}

        for change in changes:
            previous = seen.get(change.event_id)

            if previous is not None:
                return False

            seen[change.event_id] = change.action

        return True

    def _scenario_reduction(
        self,
        changes: Iterable[SpendingChange],
        events: Iterable[FinancialEvent],
    ) -> float:
        events_by_id = {
            event.event_id: event
            for event in events
        }

        total = 0.0

        for change in changes:
            event = events_by_id.get(change.event_id)

            if event is None:
                continue

            amount = self._safe_float(event.amount)

            if change.action == "stop":
                total += amount

            elif change.action == "reduce_to":
                new_amount = self._safe_float(change.new_amount)
                total += max(0.0, amount - new_amount)

        return total

    # ------------------------------------------------------------------
    # Profile compatibility
    # ------------------------------------------------------------------

    def _context_reduce_categories(
        self,
        context: UserContext,
    ) -> list[str]:
        """
        The normalized UserContext contains flexible_categories for
        compatibility with the rest of the architecture.

        If the richer fields are present, prefer them because the source
        profile distinguishes reducing from stopping.
        """

        value = getattr(
            context,
            "reduce_categories",
            None,
        )

        if value is not None:
            return list(value)

        value = getattr(
            context,
            "willing_to_reduce_categories",
            None,
        )

        if value is not None:
            return list(value)

        # Older normalized model:
        flexible = getattr(
            context,
            "flexible_categories",
            [],
        )

        stop = set(
            self._normalized_categories(
                self._context_stop_categories(context)
            )
        )

        return [
            category
            for category in flexible
            if self._normalize(category) not in stop
        ]

    def _context_stop_categories(
        self,
        context: UserContext,
    ) -> list[str]:
        value = getattr(
            context,
            "stop_categories",
            None,
        )

        if value is not None:
            return list(value)

        value = getattr(
            context,
            "willing_to_stop_categories",
            None,
        )

        if value is not None:
            return list(value)

        return []

    # ------------------------------------------------------------------
    # Formatting / helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _minimum_allowed_amount(
        event: FinancialEvent,
    ) -> float:
        minimum = getattr(
            event,
            "minimum_allowed_amount",
            None,
        )

        if minimum is None:
            # A reducible event without an explicit minimum can safely
            # be reduced to zero only when the event is also stoppable.
            # For pure reduction we conservatively preserve the current
            # amount rather than inventing a lower minimum.
            return 0.0

        return max(0.0, float(minimum))

    @staticmethod
    def _safe_float(value) -> float:
        if value is None:
            return 0.0

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _normalize(value) -> str:
        if value is None:
            return ""

        return (
            str(value)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    @classmethod
    def _normalized_categories(
        cls,
        values,
    ) -> set[str]:
        if values is None:
            return set()

        if isinstance(values, str):
            values = values.split("|")

        return {
            cls._normalize(value)
            for value in values
            if cls._normalize(value)
        }

    @staticmethod
    def _date_only(value):
        if value is None:
            return None

        if hasattr(value, "date"):
            try:
                return value.date()
            except Exception:
                pass

        text = str(value).strip()

        if not text:
            return None

        try:
            from datetime import datetime

            return datetime.fromisoformat(
                text.replace("Z", "+00:00")
            ).date()
        except ValueError:
            try:
                from datetime import date

                return date.fromisoformat(text[:10])
            except ValueError:
                return None

    @classmethod
    def _date_key(cls, value):
        parsed = cls._date_only(value)

        if parsed is None:
            return "9999-12-31"

        return parsed.isoformat()


def _format_amount(value: float) -> str:
    """
    Format an amount without unnecessary trailing zeros.

    Examples:
        1000.0   -> "1000"
        1000.50  -> "1000.5"
        1000.55  -> "1000.55"
    """

    value = float(value)

    if abs(value) < 1e-9:
        return "0"

    if value.is_integer():
        return str(int(value))

    return f"{value:.2f}".rstrip("0").rstrip(".")