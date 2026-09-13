from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Optional

from ..normalization.models import (
    FinancialRequest,
    PaymentOption,
    UserContext,
)


EPSILON = 1e-9


@dataclass(frozen=True)
class ScheduledPayment:
    date: date
    amount: float


@dataclass(frozen=True)
class PaymentPlanResult:
    """
    Result of evaluating one possible payment plan.
    """

    valid: bool
    payment_method: str
    payments: tuple[ScheduledPayment, ...]
    total_payable: float
    financing_fee: float
    payment_option_id: Optional[str] = None
    reason: str = ""

    @property
    def first_payment_date(self) -> Optional[date]:
        if not self.payments:
            return None

        return self.payments[0].date

    @property
    def last_payment_date(self) -> Optional[date]:
        if not self.payments:
            return None

        return self.payments[-1].date


class PaymentPlanEngine:
    """
    Builds and validates payment plans.

    Responsibilities:
      * validate supplied payment options
      * construct exact installment schedules
      * construct the challenge's special partial-payment schedule
      * verify payment-method permissions
      * verify installment-month limits
      * verify deadline constraints
      * verify forecast safety
      * never invent an installment schedule
    """

    def __init__(self, forecaster=None) -> None:
        self.forecaster = forecaster

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        request: FinancialRequest,
        context: UserContext,
        payment_options: Iterable[PaymentOption],
        forecast=None,
        safe_amount: Optional[float] = None,
        earliest_full_payment: Optional[date] = None,
    ) -> list[PaymentPlanResult]:
        """
        Evaluate all supplied payment options plus eligible partial payment.

        Returns only plans that satisfy the basic contractual rules.

        The decision engine is responsible for ranking the returned plans.
        """

        results: list[PaymentPlanResult] = []

        requested_amount = self._money(
            request.requested_amount
        )

        request_date = self._date_only(
            request.request_date
        )

        deadline = self._date_only(
            request.desired_completion_date
        )

        accepted_methods = self._accepted_methods(context)

        # --------------------------------------------------------------
        # FULL PAYMENT
        # --------------------------------------------------------------

        if "full_payment" in accepted_methods:
            if (
                safe_amount is not None
                and safe_amount + EPSILON >= requested_amount
            ):
                plan = PaymentPlanResult(
                    valid=True,
                    payment_method="full_payment",
                    payments=(
                        ScheduledPayment(
                            date=request_date,
                            amount=requested_amount,
                        ),
                    ),
                    total_payable=requested_amount,
                    financing_fee=0.0,
                    payment_option_id=None,
                    reason="Full payment is safe on the request date.",
                )

                if self._meets_deadline(plan, deadline):
                    if self._schedule_is_safe(
                        plan,
                        forecast,
                    ):
                        results.append(plan)

        # --------------------------------------------------------------
        # PARTIAL PAYMENT
        # --------------------------------------------------------------

        if (
            "partial_payment" in accepted_methods
            and self._allows_partial(request)
            and safe_amount is not None
            and safe_amount > EPSILON
            and safe_amount + EPSILON < requested_amount
            and earliest_full_payment is not None
        ):
            partial_date = request_date
            full_date = self._date_only(
                earliest_full_payment
            )

            # The second payment must complete the purchase by the
            # desired completion date.
            if (
                deadline is None
                or full_date <= deadline
            ):
                partial_plan = self.build_partial_payment_plan(
                    request=request,
                    safe_amount=safe_amount,
                    full_payment_date=full_date,
                )

                if partial_plan.valid:
                    if self._schedule_is_safe(
                        partial_plan,
                        forecast,
                    ):
                        results.append(partial_plan)

        # --------------------------------------------------------------
        # SUPPLIED INSTALLMENT OPTIONS
        # --------------------------------------------------------------

        for option in payment_options:
            plan = self.evaluate_payment_option(
                request=request,
                context=context,
                option=option,
                forecast=forecast,
            )

            if plan.valid:
                results.append(plan)

        return self._deduplicate(results)

    def evaluate_payment_option(
        self,
        request: FinancialRequest,
        context: UserContext,
        option: PaymentOption,
        forecast=None,
    ) -> PaymentPlanResult:
        """
        Validate one supplied payment option.

        IMPORTANT:
        We never invent an installment frequency, number of payments,
        amount, fee, or date. Everything comes directly from the supplied
        request_payment_options.csv row.
        """

        payment_method = self._normalize_method(
            option.payment_method
        )

        accepted_methods = self._accepted_methods(context)

        if payment_method not in accepted_methods:
            return self._invalid(
                payment_method=payment_method,
                option_id=option.payment_option_id,
                reason=(
                    f"User does not consider payment method "
                    f"{payment_method}."
                ),
            )

        if payment_method != "installments":
            return self._invalid(
                payment_method=payment_method,
                option_id=option.payment_option_id,
                reason=(
                    "Supplied payment option is not an installment "
                    "option."
                ),
            )

        number_of_payments = self._safe_int(
            option.number_of_payments
        )

        if number_of_payments <= 0:
            return self._invalid(
                payment_method=payment_method,
                option_id=option.payment_option_id,
                reason="Installment count must be positive.",
            )

        max_months = self._safe_int(
            getattr(
                context,
                "max_installment_months",
                None,
            )
        )

        if max_months > 0:
            last_date = self._option_last_payment_date(
                option
            )

            request_date = self._date_only(
                request.request_date
            )

            if (
                last_date is not None
                and request_date is not None
            ):
                months = self._months_between(
                    request_date,
                    last_date,
                )

                if months > max_months:
                    return self._invalid(
                        payment_method=payment_method,
                        option_id=option.payment_option_id,
                        reason=(
                            "Installment plan exceeds the user's "
                            "maximum installment duration."
                        ),
                    )

        payments = self._build_exact_option_schedule(
            option
        )

        if not payments:
            return self._invalid(
                payment_method=payment_method,
                option_id=option.payment_option_id,
                reason="Could not construct installment schedule.",
            )

        # The supplied schedule must have the declared number of
        # payments.
        if len(payments) != number_of_payments:
            return self._invalid(
                payment_method=payment_method,
                option_id=option.payment_option_id,
                reason=(
                    "Supplied installment schedule does not match "
                    "number_of_payments."
                ),
            )

        request_amount = self._money(
            request.requested_amount
        )

        expected_total = self._money(
            option.total_payable
        )

        if expected_total <= 0:
            expected_total = sum(
                payment.amount
                for payment in payments
            )

        actual_total = sum(
            payment.amount
            for payment in payments
        )

        # Allow tiny floating-point differences.
        if abs(actual_total - expected_total) > 0.02:
            return self._invalid(
                payment_method=payment_method,
                option_id=option.payment_option_id,
                reason=(
                    "Installment schedule does not match the supplied "
                    "total payable amount."
                ),
            )

        # An installment option must actually cover the requested
        # amount.
        if actual_total + 0.02 < request_amount:
            return self._invalid(
                payment_method=payment_method,
                option_id=option.payment_option_id,
                reason=(
                    "Installment schedule does not fully cover the "
                    "requested amount."
                ),
            )

        financing_fee = self._money(
            option.financing_fee
        )

        deadline = self._date_only(
            request.desired_completion_date
        )

        last_payment = payments[-1].date

        if (
            deadline is not None
            and last_payment > deadline
        ):
            return self._invalid(
                payment_method=payment_method,
                option_id=option.payment_option_id,
                reason=(
                    "Installment plan completes after the requested "
                    "completion date."
                ),
            )

        plan = PaymentPlanResult(
            valid=True,
            payment_method="installments",
            payments=tuple(payments),
            total_payable=expected_total,
            financing_fee=financing_fee,
            payment_option_id=option.payment_option_id,
            reason="Supplied installment option is valid.",
        )

        if not self._schedule_is_safe(
            plan,
            forecast,
        ):
            return self._invalid(
                payment_method=payment_method,
                option_id=option.payment_option_id,
                reason=(
                    "The installment schedule would violate the "
                    "projected minimum balance."
                ),
            )

        return plan

    def build_partial_payment_plan(
        self,
        request: FinancialRequest,
        safe_amount: float,
        full_payment_date: date,
    ) -> PaymentPlanResult:
        """
        Build the challenge-specific partial-payment plan.

        Exactly two payments:

          request_date : safe_amount
          full_payment_date : requested_amount - safe_amount

        No financing fee is invented.
        """

        request_date = self._date_only(
            request.request_date
        )

        requested_amount = self._money(
            request.requested_amount
        )

        safe_amount = self._money(
            safe_amount
        )

        full_payment_date = self._date_only(
            full_payment_date
        )

        if request_date is None:
            return self._invalid(
                payment_method="partial_payment",
                reason="Request date is missing.",
            )

        if full_payment_date is None:
            return self._invalid(
                payment_method="partial_payment",
                reason="Full-payment date is missing.",
            )

        if safe_amount <= 0:
            return self._invalid(
                payment_method="partial_payment",
                reason="Safe partial amount must be positive.",
            )

        if safe_amount >= requested_amount:
            return self._invalid(
                payment_method="partial_payment",
                reason=(
                    "Partial payment must be less than the "
                    "requested amount."
                ),
            )

        if full_payment_date < request_date:
            return self._invalid(
                payment_method="partial_payment",
                reason=(
                    "Full-payment date cannot precede request date."
                ),
            )

        remaining = self._money(
            requested_amount - safe_amount
        )

        if remaining <= 0:
            return self._invalid(
                payment_method="partial_payment",
                reason="Remaining payment must be positive.",
            )

        payments = (
            ScheduledPayment(
                date=request_date,
                amount=safe_amount,
            ),
            ScheduledPayment(
                date=full_payment_date,
                amount=remaining,
            ),
        )

        return PaymentPlanResult(
            valid=True,
            payment_method="partial_payment",
            payments=payments,
            total_payable=requested_amount,
            financing_fee=0.0,
            payment_option_id=None,
            reason=(
                "Two-payment partial plan using the safe amount "
                "on the request date and the remaining balance "
                "on the earliest safe full-payment date."
            ),
        )

    def build_full_payment_plan(
        self,
        request: FinancialRequest,
    ) -> PaymentPlanResult:
        """
        Construct a simple full-payment plan.

        Safety is intentionally checked elsewhere because this method
        only constructs the schedule.
        """

        request_date = self._date_only(
            request.request_date
        )

        amount = self._money(
            request.requested_amount
        )

        return PaymentPlanResult(
            valid=True,
            payment_method="full_payment",
            payments=(
                ScheduledPayment(
                    date=request_date,
                    amount=amount,
                ),
            ),
            total_payable=amount,
            financing_fee=0.0,
            payment_option_id=None,
            reason="Full payment on the request date.",
        )

    def format_plan(
        self,
        plan: PaymentPlanResult,
    ) -> str:
        """
        Convert a plan to the exact output representation:

            YYYY-MM-DD:amount|YYYY-MM-DD:amount

        Empty plans become 'none'.
        """

        if not plan.valid or not plan.payments:
            return "none"

        return "|".join(
            f"{payment.date.isoformat()}:{_format_amount(payment.amount)}"
            for payment in plan.payments
        )

    # ------------------------------------------------------------------
    # Exact installment construction
    # ------------------------------------------------------------------

    def _build_exact_option_schedule(
        self,
        option: PaymentOption,
    ) -> list[ScheduledPayment]:
        number_of_payments = self._safe_int(
            option.number_of_payments
        )

        first_date = self._date_only(
            option.first_payment_date
        )

        amount = self._money(
            option.payment_amount
        )

        frequency_days = self._safe_int(
            option.payment_frequency_days
        )

        if (
            number_of_payments <= 0
            or first_date is None
            or amount <= 0
        ):
            return []

        payments: list[ScheduledPayment] = []

        for index in range(number_of_payments):
            if index == 0:
                payment_date = first_date
            else:
                payment_date = first_date.fromordinal(
                    first_date.toordinal()
                    + (
                        frequency_days * index
                    )
                )

            payments.append(
                ScheduledPayment(
                    date=payment_date,
                    amount=amount,
                )
            )

        return payments

    def _option_last_payment_date(
        self,
        option: PaymentOption,
    ) -> Optional[date]:
        payments = self._build_exact_option_schedule(
            option
        )

        if not payments:
            return None

        return payments[-1].date

    # ------------------------------------------------------------------
    # Forecast safety
    # ------------------------------------------------------------------

    def _schedule_is_safe(
        self,
        plan: PaymentPlanResult,
        forecast,
    ) -> bool:
        """
        Check whether every payment can be made without violating the
        projected minimum balance.

        The forecaster interface may expose:
          * can_afford_schedule(...)
          * is_schedule_safe(...)
          * minimum_balance_after(...)
          * daily balances

        We support these interfaces so the payment engine remains
        compatible with the forecast implementation.
        """

        if forecast is None:
            # Do not reject a plan merely because the caller chose not
            # to provide a forecast. The affordability engine is still
            # responsible for the baseline decision.
            return True

        payments = [
            {
                "date": payment.date,
                "amount": payment.amount,
            }
            for payment in plan.payments
        ]

        # Preferred interface.
        method = getattr(
            forecast,
            "can_afford_schedule",
            None,
        )

        if callable(method):
            try:
                return bool(
                    method(payments)
                )
            except TypeError:
                try:
                    return bool(
                        method(
                            payments=payments
                        )
                    )
                except Exception:
                    pass
            except Exception:
                pass

        # Alternative interface.
        method = getattr(
            forecast,
            "is_schedule_safe",
            None,
        )

        if callable(method):
            try:
                return bool(
                    method(payments)
                )
            except TypeError:
                try:
                    return bool(
                        method(
                            payments=payments
                        )
                    )
                except Exception:
                    pass
            except Exception:
                pass

        # If the forecast exposes a balance lookup, simulate the
        # additional payments against the projected balance.
        balance_lookup = getattr(
            forecast,
            "balance_on",
            None,
        )

        minimum_balance = getattr(
            forecast,
            "minimum_balance_to_keep",
            None,
        )

        if (
            callable(balance_lookup)
            and minimum_balance is not None
        ):
            for payment in plan.payments:
                try:
                    balance = float(
                        balance_lookup(
                            payment.date
                        )
                    )

                    if (
                        balance - payment.amount
                        < float(minimum_balance) - 0.01
                    ):
                        return False

                except Exception:
                    return True

        # No recognized forecast interface. Leave the decision to the
        # main affordability engine instead of inventing a result.
        return True

    # ------------------------------------------------------------------
    # User preferences
    # ------------------------------------------------------------------

    def _accepted_methods(
        self,
        context: UserContext,
    ) -> set[str]:
        methods = getattr(
            context,
            "payment_methods_user_will_consider",
            [],
        )

        if isinstance(methods, str):
            methods = methods.split("|")

        normalized = set()

        for method in methods:
            value = self._normalize_method(
                method
            )

            if value:
                normalized.add(value)

        return normalized

    @staticmethod
    def _allows_partial(
        request: FinancialRequest,
    ) -> bool:
        value = getattr(
            request,
            "allows_partial_payment",
            False,
        )

        if isinstance(value, bool):
            return value

        return str(value).strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _meets_deadline(
        plan: PaymentPlanResult,
        deadline: Optional[date],
    ) -> bool:
        if deadline is None:
            return True

        if not plan.payments:
            return False

        return plan.payments[-1].date <= deadline

    @staticmethod
    def _invalid(
        payment_method: str,
        option_id: Optional[str] = None,
        reason: str = "",
    ) -> PaymentPlanResult:
        return PaymentPlanResult(
            valid=False,
            payment_method=payment_method,
            payments=tuple(),
            total_payable=0.0,
            financing_fee=0.0,
            payment_option_id=option_id,
            reason=reason,
        )

    @staticmethod
    def _deduplicate(
        plans: Iterable[PaymentPlanResult],
    ) -> list[PaymentPlanResult]:
        result: list[PaymentPlanResult] = []
        seen: set[tuple] = set()

        for plan in plans:
            key = (
                plan.payment_method,
                plan.payment_option_id,
                tuple(
                    (
                        payment.date,
                        round(payment.amount, 8),
                    )
                    for payment in plan.payments
                ),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(plan)

        return result

    # ------------------------------------------------------------------
    # Generic parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _money(value) -> float:
        if value is None:
            return 0.0

        try:
            return round(float(value), 2)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_int(value) -> int:
        if value is None:
            return 0

        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _normalize_method(value) -> str:
        if value is None:
            return ""

        value = (
            str(value)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        aliases = {
            "full": "full_payment",
            "fullpayment": "full_payment",
            "full_payment": "full_payment",
            "partial": "partial_payment",
            "partialpayment": "partial_payment",
            "partial_payment": "partial_payment",
            "installment": "installments",
            "installments": "installments",
            "wait": "wait",
            "not_recommended": "not_recommended",
        }

        return aliases.get(
            value,
            value,
        )

    @staticmethod
    def _date_only(value) -> Optional[date]:
        if value is None:
            return None

        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, date):
            return value

        text = str(value).strip()

        if not text:
            return None

        try:
            return datetime.fromisoformat(
                text.replace("Z", "+00:00")
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
    def _months_between(
        start: date,
        end: date,
    ) -> int:
        if end <= start:
            return 0

        months = (
            (end.year - start.year) * 12
            + (end.month - start.month)
        )

        if end.day > start.day:
            months += 1

        return months


def _format_amount(value: float) -> str:
    value = round(float(value), 2)

    if abs(value) < 0.005:
        return "0"

    if value.is_integer():
        return str(int(value))

    return f"{value:.2f}".rstrip("0").rstrip(".")