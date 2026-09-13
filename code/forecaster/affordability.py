from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Sequence

from code.forecaster.cash_flow import CashFlowForecaster, ForecastResult
from code.normalization.models import FinancialEvent, FinancialRequest, UserContext


@dataclass
class AffordabilityResult:
    requested_amount: float
    amount_safe_to_pay: float
    earliest_date_for_full_payment: Optional[date]

    forecast: ForecastResult

    safe_today: bool
    safe_by_deadline: bool

    spending_overrides: Dict[str, float]

    @property
    def full_amount_is_safe(self) -> bool:
        return (
            self.earliest_date_for_full_payment is not None
        )

    @property
    def status_without_payment_preferences(self) -> str:
        """
        Financial capacity only.

        Payment-method preferences are intentionally handled later by the
        decision policy because earliest_date_for_full_payment is independent
        of the user's preferred payment methods.
        """
        if self.safe_today:
            return "affordable_now"

        if self.safe_by_deadline:
            return "affordable_later"

        return "not_affordable"


class AffordabilityEngine:
    """
    Computes affordability independently from payment-method preferences.

    This separation is important because the official specification says
    earliest_date_for_full_payment measures financial capacity independently
    of whether the user accepts full_payment.
    """

    def __init__(
        self,
        forecaster: CashFlowForecaster,
    ) -> None:
        self.forecaster = forecaster

    def analyze(
        self,
        user: UserContext,
        request: FinancialRequest,
        events: Sequence[FinancialEvent],
        spending_overrides: Optional[Dict[str, float]] = None,
    ) -> AffordabilityResult:
        requested_amount = max(
            0.0,
            float(request.requested_amount),
        )

        overrides = spending_overrides or {}

        forecast = self.forecaster.build_forecast(
            user=user,
            events=events,
            start_date=request.request_date,
            horizon_days=self.forecaster.FORECAST_DAYS,
            spending_overrides=overrides,
        )

        safe_amount = self.forecaster.max_safe_payment_today(
            user=user,
            events=events,
            request_date=request.request_date,
            requested_amount=requested_amount,
            spending_overrides=overrides,
        )

        earliest_date = (
            self.forecaster.earliest_full_payment_date(
                user=user,
                events=events,
                request_date=request.request_date,
                requested_amount=requested_amount,
                spending_overrides=overrides,
            )
        )

        safe_today = (
            requested_amount <= safe_amount + 1e-7
        )

        safe_by_deadline = (
            earliest_date is not None
            and earliest_date <= request.desired_completion_date
        )

        return AffordabilityResult(
            requested_amount=round(requested_amount, 2),
            amount_safe_to_pay=round(
                min(requested_amount, max(0.0, safe_amount)),
                2,
            ),
            earliest_date_for_full_payment=earliest_date,
            forecast=forecast,
            safe_today=safe_today,
            safe_by_deadline=safe_by_deadline,
            spending_overrides=dict(overrides),
        )

    def analyze_with_spending_changes(
        self,
        user: UserContext,
        request: FinancialRequest,
        events: Sequence[FinancialEvent],
        spending_overrides: Dict[str, float],
    ) -> AffordabilityResult:
        """
        Convenience wrapper for evaluating a candidate spending-change plan.
        """
        return self.analyze(
            user=user,
            request=request,
            events=events,
            spending_overrides=spending_overrides,
        )

    @staticmethod
    def remaining_after_safe_payment(
        result: AffordabilityResult,
    ) -> float:
        """
        Conservative amount remaining at the worst projected point after
        paying the currently safe amount.
        """
        return round(
            result.forecast.minimum_projected_balance
            - result.amount_safe_to_pay,
            2,
        )