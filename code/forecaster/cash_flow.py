from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

from code.normalization.models import FinancialEvent, UserContext


@dataclass(frozen=True)
class ForecastPoint:
    date: date
    balance: float


@dataclass
class ForecastResult:
    start_date: date
    end_date: date
    starting_balance: float
    minimum_balance: float
    points: List[ForecastPoint]

    @property
    def minimum_projected_balance(self) -> float:
        if not self.points:
            return self.starting_balance

        return min(point.balance for point in self.points)

    @property
    def is_safe(self) -> bool:
        return self.minimum_projected_balance >= self.minimum_balance

    def balance_on(self, target_date: date) -> Optional[float]:
        for point in self.points:
            if point.date == target_date:
                return point.balance

        return None

    def first_safe_date(
        self,
        required_amount: float,
        from_date: Optional[date] = None,
    ) -> Optional[date]:
        """
        Find the first date on which paying required_amount on that date
        remains safe for the remainder of the forecast.

        This is intentionally stronger than checking only the balance on
        that date: a payment is not considered safe if it creates a later
        minimum-balance violation.
        """
        if required_amount < 0:
            return None

        start = from_date or self.start_date

        for point in self.points:
            if point.date < start:
                continue

            if self._payment_is_safe_on_date(
                required_amount,
                point.date,
            ):
                return point.date

        return None

    def _payment_is_safe_on_date(
        self,
        amount: float,
        payment_date: date,
    ) -> bool:
        """
        Reconstruct safety after subtracting a one-time payment from the
        balance on payment_date and every subsequent day.
        """
        for point in self.points:
            if point.date < payment_date:
                continue

            adjusted_balance = point.balance - amount

            if adjusted_balance < self.minimum_balance - 1e-9:
                return False

        return True


class CashFlowForecaster:
    """
    Build a deterministic 90-day cash-flow forecast.

    The forecast:
    - starts from the user's supplied available balance;
    - protects minimum_balance_to_keep;
    - reserves pending debits;
    - ignores pending/failed/cancelled credits;
    - recognizes confirmed future credits on settlement date;
    - includes supplied future events;
    - cautiously extrapolates strongly recurring events when future rows
      are not already supplied.
    """

    FORECAST_DAYS = 90
    EPSILON = 1e-9

    ACTIVE_STATUSES = {
        "settled",
        "scheduled",
        "pending",
        "confirmed",
        "posted",
        "authorized",
    }

    EXCLUDED_STATUSES = {
        "cancelled",
        "canceled",
        "failed",
        "reversed",
        "declined",
        "expired",
    }

    CREDIT_ALLOWED_STATUSES = {
        "settled",
        "scheduled",
        "confirmed",
        "posted",
    }

    def __init__(
        self,
        currency_converter: Optional[
            Callable[[float, str, str, date], Optional[float]]
        ] = None,
    ) -> None:
        self.currency_converter = currency_converter

    def build_forecast(
        self,
        user: UserContext,
        events: Sequence[FinancialEvent],
        start_date: date,
        horizon_days: int = FORECAST_DAYS,
        spending_overrides: Optional[Dict[str, float]] = None,
    ) -> ForecastResult:
        horizon_days = max(0, int(horizon_days))

        end_date = start_date + timedelta(days=horizon_days)

        overrides = spending_overrides or {}

        relevant_events = self._prepare_events(
            user=user,
            events=events,
            start_date=start_date,
            end_date=end_date,
            spending_overrides=overrides,
        )

        daily_net: Dict[date, float] = {}

        for event in relevant_events:
            event_date = self._effective_event_date(event)

            if event_date < start_date or event_date > end_date:
                continue

            amount = self._amount_in_home_currency(
                event=event,
                user=user,
            )

            if amount is None:
                continue

            if amount < 0:
                continue

            direction = str(event.direction or "").strip().lower()

            if direction == "credit":
                daily_net[event_date] = (
                    daily_net.get(event_date, 0.0) + amount
                )
            elif direction == "debit":
                daily_net[event_date] = (
                    daily_net.get(event_date, 0.0) - amount
                )

        points: List[ForecastPoint] = []

        balance = float(user.current_available_balance)

        current = start_date

        while current <= end_date:
            balance += daily_net.get(current, 0.0)

            points.append(
                ForecastPoint(
                    date=current,
                    balance=round(balance, 10),
                )
            )

            current += timedelta(days=1)

        return ForecastResult(
            start_date=start_date,
            end_date=end_date,
            starting_balance=float(user.current_available_balance),
            minimum_balance=float(user.minimum_balance_to_keep),
            points=points,
        )

    def max_safe_payment_today(
        self,
        user: UserContext,
        events: Sequence[FinancialEvent],
        request_date: date,
        requested_amount: float,
        spending_overrides: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Find the largest one-time amount payable on request_date while
        preserving the minimum balance throughout the forecast.

        Binary search is used because the safety condition is monotonic:
        if amount X is safe, every amount smaller than X is also safe.
        """
        requested_amount = max(0.0, float(requested_amount))

        forecast = self.build_forecast(
            user=user,
            events=events,
            start_date=request_date,
            horizon_days=self.FORECAST_DAYS,
            spending_overrides=spending_overrides,
        )

        baseline_minimum = forecast.minimum_projected_balance

        available_for_request = max(
            0.0,
            baseline_minimum - forecast.minimum_balance,
        )

        upper = min(
            requested_amount,
            available_for_request,
        )

        if upper <= self.EPSILON:
            return 0.0

        low = 0.0
        high = upper

        for _ in range(60):
            mid = (low + high) / 2.0

            if self._payment_is_safe(
                forecast=forecast,
                amount=mid,
                payment_date=request_date,
            ):
                low = mid
            else:
                high = mid

        return round(
            max(0.0, min(requested_amount, low)),
            2,
        )

    def earliest_full_payment_date(
        self,
        user: UserContext,
        events: Sequence[FinancialEvent],
        request_date: date,
        requested_amount: float,
        spending_overrides: Optional[Dict[str, float]] = None,
    ) -> Optional[date]:
        """
        Find the first date within the 90-day forecast on which the full
        requested amount can safely be paid.
        """
        requested_amount = max(0.0, float(requested_amount))

        forecast = self.build_forecast(
            user=user,
            events=events,
            start_date=request_date,
            horizon_days=self.FORECAST_DAYS,
            spending_overrides=spending_overrides,
        )

        return forecast.first_safe_date(
            required_amount=requested_amount,
            from_date=request_date,
        )

    def _payment_is_safe(
        self,
        forecast: ForecastResult,
        amount: float,
        payment_date: date,
    ) -> bool:
        for point in forecast.points:
            if point.date < payment_date:
                continue

            if point.balance - amount < (
                forecast.minimum_balance - self.EPSILON
            ):
                return False

        return True

    def _prepare_events(
        self,
        user: UserContext,
        events: Sequence[FinancialEvent],
        start_date: date,
        end_date: date,
        spending_overrides: Dict[str, float],
    ) -> List[FinancialEvent]:
        """
        Select valid future events and cautiously add recurring projections.
        """
        prepared: List[FinancialEvent] = []

        existing_future_keys: Set[str] = set()

        for event in events:
            if not self._event_can_affect_forecast(event):
                continue

            effective_date = self._effective_event_date(event)

            if effective_date < start_date:
                continue

            if effective_date > end_date:
                continue

            if (
                event.event_id
                and event.event_id in spending_overrides
            ):
                amount = spending_overrides[event.event_id]

                event = self._copy_with_amount(
                    event,
                    amount,
                )

            prepared.append(event)

            if event.event_id:
                existing_future_keys.add(event.event_id)

        projected = self._project_recurring_events(
            events=events,
            start_date=start_date,
            end_date=end_date,
            existing_event_ids=existing_future_keys,
            spending_overrides=spending_overrides,
        )

        prepared.extend(projected)

        return prepared

    def _event_can_affect_forecast(
        self,
        event: FinancialEvent,
    ) -> bool:
        status = str(event.status or "").strip().lower()

        if status in self.EXCLUDED_STATUSES:
            return False

        direction = str(event.direction or "").strip().lower()

        if direction not in {"credit", "debit"}:
            return False

        if event.amount is None:
            return False

        # Pending credits must never be counted.
        if direction == "credit":
            if status not in self.CREDIT_ALLOWED_STATUSES:
                return False

        # Pending debits ARE deliberately retained.
        if direction == "debit":
            if status not in self.ACTIVE_STATUSES:
                return False

        return True

    @staticmethod
    def _effective_event_date(event: FinancialEvent) -> date:
        """
        Debits use settlement_date when supplied because that is when the
        cash impact is expected to settle. Credits also use settlement_date.
        """
        if event.settlement_date:
            return event.settlement_date

        return event.event_date

    def _amount_in_home_currency(
        self,
        event: FinancialEvent,
        user: UserContext,
    ) -> Optional[float]:
        if event.amount is None:
            return None

        source_currency = str(event.currency or "").upper()
        target_currency = str(user.home_currency or "").upper()

        if not source_currency or not target_currency:
            return None

        amount = float(event.amount)

        if source_currency == target_currency:
            return amount

        if self.currency_converter is None:
            return None

        event_date = self._effective_event_date(event)

        try:
            return self.currency_converter(
                amount,
                source_currency,
                target_currency,
                event_date,
            )
        except Exception:
            return None

    def _project_recurring_events(
        self,
        events: Sequence[FinancialEvent],
        start_date: date,
        end_date: date,
        existing_event_ids: Set[str],
        spending_overrides: Dict[str, float],
    ) -> List[FinancialEvent]:
        """
        Extrapolate only high-confidence recurring streams.

        Requirements:
        - at least 3 historical occurrences;
        - consistent direction/category/description;
        - reasonably stable intervals;
        - reasonably stable amounts;
        - no existing future row on the projected date.

        This prevents one-off purchases from accidentally becoming recurring.
        """
        groups: Dict[str, List[FinancialEvent]] = {}

        for event in events:
            if event.amount is None:
                continue

            if not event.event_date:
                continue

            status = str(event.status or "").lower()

            if status not in {
                "settled",
                "posted",
                "confirmed",
            }:
                continue

            key = self._recurrence_key(event)

            groups.setdefault(key, []).append(event)

        projections: List[FinancialEvent] = []

        for group in groups.values():
            if len(group) < 3:
                continue

            group = sorted(
                group,
                key=lambda item: item.event_date,
            )

            interval_days = self._stable_interval(group)

            if interval_days is None:
                continue

            amounts = [
                float(event.amount)
                for event in group
                if event.amount is not None
            ]

            if not amounts:
                continue

            typical_amount = median(amounts)

            if not self._amounts_are_stable(amounts, typical_amount):
                continue

            last = group[-1]

            next_date = last.event_date + timedelta(
                days=interval_days
            )

            while next_date <= end_date:
                if next_date >= start_date:
                    duplicate = any(
                        self._effective_event_date(existing) == next_date
                        and self._recurrence_key(existing)
                        == self._recurrence_key(last)
                        for existing in events
                    )

                    if not duplicate:
                        synthetic_id = (
                            f"forecast:{last.event_id}:{next_date.isoformat()}"
                        )

                        if synthetic_id not in existing_event_ids:
                            projected = self._copy_with_date(
                                last,
                                synthetic_id,
                                next_date,
                                typical_amount,
                            )

                            if projected.event_id in spending_overrides:
                                projected = self._copy_with_amount(
                                    projected,
                                    spending_overrides[
                                        projected.event_id
                                    ],
                                )

                            projections.append(projected)

                next_date += timedelta(
                    days=interval_days
                )

        return projections

    @staticmethod
    def _recurrence_key(event: FinancialEvent) -> str:
        return "|".join(
            [
                str(event.user_id or ""),
                str(event.event_type or ""),
                str(event.description or "").strip().lower(),
                str(event.category or "").strip().lower(),
                str(event.direction or "").strip().lower(),
                str(event.currency or "").upper(),
            ]
        )

    @staticmethod
    def _stable_interval(
        events: Sequence[FinancialEvent],
    ) -> Optional[int]:
        intervals: List[int] = []

        for previous, current in zip(
            events,
            events[1:],
        ):
            if previous.event_date is None:
                continue

            if current.event_date is None:
                continue

            days = (
                current.event_date - previous.event_date
            ).days

            if days <= 0 or days > 120:
                continue

            intervals.append(days)

        if len(intervals) < 2:
            return None

        typical = median(intervals)

        # Allow small calendar irregularities but reject highly irregular
        # transaction streams.
        for interval in intervals:
            if abs(interval - typical) > max(3, typical * 0.15):
                return None

        return max(1, int(round(typical)))

    @staticmethod
    def _amounts_are_stable(
        amounts: Sequence[float],
        typical: float,
    ) -> bool:
        if typical <= 0:
            return False

        deviations = [
            abs(amount - typical) / typical
            for amount in amounts
        ]

        # Recurring utilities and similar expenses can fluctuate. Permit
        # meaningful variation but reject highly heterogeneous groups.
        return max(deviations) <= 0.35

    @staticmethod
    def _copy_with_amount(
        event: FinancialEvent,
        amount: float,
    ) -> FinancialEvent:
        values = event.__dict__.copy()
        values["amount"] = float(amount)

        return FinancialEvent(**values)

    @staticmethod
    def _copy_with_date(
        event: FinancialEvent,
        event_id: str,
        event_date: date,
        amount: float,
    ) -> FinancialEvent:
        values = event.__dict__.copy()

        values["event_id"] = event_id
        values["event_date"] = event_date
        values["settlement_date"] = event_date
        values["amount"] = float(amount)
        values["status"] = "scheduled"

        return FinancialEvent(**values)