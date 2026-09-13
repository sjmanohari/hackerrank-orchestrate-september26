from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

from ..financial.payment_plans import (
    PaymentPlanEngine,
    PaymentPlanResult,
    ScheduledPayment,
)
from ..financial.spending_adjustments import (
    AdjustmentScenario,
    SpendingAdjustmentEngine,
    SpendingChange,
)
from ..normalization.models import (
    FinancialRequest,
    FinancialEvent,
    PaymentOption,
    UserContext,
)


EPSILON = 0.01


@dataclass(frozen=True)
class DecisionCandidate:
    """
    One possible answer to a purchase request.

    The policy evaluates many candidates and selects exactly one.
    """

    affordability_status: str
    payment_method: str
    amount_safe_to_pay: float
    payment_plan: str
    earliest_date_for_full_payment: Optional[date]
    spending_changes: tuple[SpendingChange, ...]
    payment_plan_result: Optional[PaymentPlanResult]

    # Ranking information.
    total_cost: float
    first_payment_date: Optional[date]
    number_of_payments: int
    payment_option_id: str
    requires_spending_changes: bool
    completes_by_deadline: bool

    # Human-readable internal reason.
    reason: str = ""


@dataclass(frozen=True)
class Decision:
    """
    Final output consumed by orchestration.py.
    """

    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str

    # Extra internal information useful for explanations/debugging.
    payment_option_id: str = ""
    total_cost: float = 0.0
    reason: str = ""


class DecisionPolicy:
    """
    Deterministic financial decision policy.

    High-level order:

        1. Establish baseline affordability.
        2. Consider full payment now.
        3. Consider partial payment.
        4. Consider supplied installment options.
        5. Consider permitted spending changes.
        6. Consider waiting until affordable.
        7. Otherwise return not_affordable.

    The policy never invents:
        * future income
        * payment options
        * installment schedules
        * spending permissions
        * unsupported discounts
        * future refunds or bonuses
    """

    def __init__(
        self,
        affordability_engine,
        payment_plan_engine: Optional[PaymentPlanEngine] = None,
        spending_adjustment_engine: Optional[
            SpendingAdjustmentEngine
        ] = None,
    ) -> None:
        self.affordability_engine = affordability_engine

        self.payment_plan_engine = (
            payment_plan_engine
            or PaymentPlanEngine()
        )

        self.spending_adjustment_engine = (
            spending_adjustment_engine
            or SpendingAdjustmentEngine()
        )

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def decide(
        self,
        request: FinancialRequest,
        context: UserContext,
        events: Iterable[FinancialEvent],
        payment_options: Iterable[PaymentOption],
        forecast=None,
        baseline_affordability=None,
    ) -> Decision:
        """
        Produce the final deterministic decision for one request.
        """

        events = list(events)
        payment_options = list(payment_options)

        request_date = self._date_only(
            request.request_date
        )

        deadline = self._date_only(
            request.desired_completion_date
        )

        requested_amount = self._money(
            request.requested_amount
        )

        # --------------------------------------------------------------
        # 1. Determine baseline affordability.
        # --------------------------------------------------------------

        affordability = (
            baseline_affordability
            or self._calculate_affordability(
                request=request,
                context=context,
                events=events,
                forecast=forecast,
            )
        )

        safe_today = self._extract_safe_amount(
            affordability
        )

        earliest_full_date = (
            self._extract_earliest_full_payment_date(
                affordability
            )
        )

        # --------------------------------------------------------------
        # 2. Generate no-change payment candidates.
        # --------------------------------------------------------------

        candidates: list[DecisionCandidate] = []

        candidates.extend(
            self._baseline_candidates(
                request=request,
                context=context,
                payment_options=payment_options,
                forecast=forecast,
                safe_today=safe_today,
                earliest_full_date=earliest_full_date,
            )
        )

        # --------------------------------------------------------------
        # 3. Generate spending-change scenarios.
        # --------------------------------------------------------------

        scenarios = (
            self.spending_adjustment_engine.generate_scenarios(
                events=events,
                context=context,
                as_of_date=request_date,
                max_changes=3,
            )
        )

        # Empty scenario was already represented by the baseline
        # candidates. We only need actual modifications here.
        for scenario in scenarios:
            if not scenario.changes:
                continue

            scenario_events = (
                self.spending_adjustment_engine.apply_changes(
                    events=events,
                    changes=scenario.changes,
                )
            )

            scenario_forecast = self._reforecast(
                request=request,
                context=context,
                events=scenario_events,
            )

            scenario_affordability = (
                self._calculate_affordability(
                    request=request,
                    context=context,
                    events=scenario_events,
                    forecast=scenario_forecast,
                )
            )

            scenario_safe_today = (
                self._extract_safe_amount(
                    scenario_affordability
                )
            )

            scenario_earliest_date = (
                self._extract_earliest_full_payment_date(
                    scenario_affordability
                )
            )

            scenario_candidates = (
                self._baseline_candidates(
                    request=request,
                    context=context,
                    payment_options=payment_options,
                    forecast=scenario_forecast,
                    safe_today=scenario_safe_today,
                    earliest_full_date=scenario_earliest_date,
                    spending_changes=scenario.changes,
                )
            )

            candidates.extend(
                scenario_candidates
            )

        # --------------------------------------------------------------
        # 4. If something is affordable, rank it.
        # --------------------------------------------------------------

        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate.completes_by_deadline
        ]

        if valid_candidates:
            best = self._select_best_candidate(
                valid_candidates,
                deadline=deadline,
            )

            return self._candidate_to_decision(
                best
            )

        # --------------------------------------------------------------
        # 5. Nothing can safely complete by deadline.
        #
        # Try a wait-only outcome if the forecast gives a future
        # affordability date.
        # --------------------------------------------------------------

        if (
            earliest_full_date is not None
            and (
                deadline is None
                or earliest_full_date <= deadline
            )
        ):
            return Decision(
                amount_safe_to_pay=0.0,
                affordability_status="affordable_later",
                recommended_payment_method="wait",
                payment_plan="none",
                earliest_date_for_full_payment=(
                    earliest_full_date.isoformat()
                ),
                spending_changes_needed="none",
                payment_option_id="",
                total_cost=requested_amount,
                reason=(
                    "The requested amount is not safely payable "
                    "today but is projected to become affordable "
                    "later."
                ),
            )

        # --------------------------------------------------------------
        # 6. Nothing works safely.
        # --------------------------------------------------------------

        return Decision(
            amount_safe_to_pay=0.0,
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment=(
                earliest_full_date.isoformat()
                if earliest_full_date is not None
                else ""
            ),
            spending_changes_needed="none",
            payment_option_id="",
            total_cost=0.0,
            reason=(
                "No supported payment strategy can safely satisfy "
                "the request within the available forecast."
            ),
        )

    # ==================================================================
    # BASELINE CANDIDATES
    # ==================================================================

    def _baseline_candidates(
        self,
        request: FinancialRequest,
        context: UserContext,
        payment_options: list[PaymentOption],
        forecast,
        safe_today: float,
        earliest_full_date: Optional[date],
        spending_changes: Iterable[SpendingChange] = (),
    ) -> list[DecisionCandidate]:
        """
        Build all payment candidates for a particular financial state.
        """

        spending_changes = tuple(
            spending_changes
        )

        candidates: list[DecisionCandidate] = []

        requested_amount = self._money(
            request.requested_amount
        )

        request_date = self._date_only(
            request.request_date
        )

        deadline = self._date_only(
            request.desired_completion_date
        )

        # --------------------------------------------------------------
        # FULL PAYMENT
        # --------------------------------------------------------------

        if (
            safe_today + EPSILON
            >= requested_amount
            and self._method_allowed(
                context,
                "full_payment",
            )
        ):
            plan = (
                self.payment_plan_engine
                .build_full_payment_plan(request)
            )

            if self._plan_safe(
                plan,
                forecast,
            ):
                candidates.append(
                    DecisionCandidate(
                        affordability_status=(
                            "affordable_now"
                        ),
                        payment_method="full_payment",
                        amount_safe_to_pay=(
                            requested_amount
                        ),
                        payment_plan=(
                            self.payment_plan_engine
                            .format_plan(plan)
                        ),
                        earliest_date_for_full_payment=(
                            request_date
                        ),
                        spending_changes=(
                            spending_changes
                        ),
                        payment_plan_result=plan,
                        total_cost=requested_amount,
                        first_payment_date=request_date,
                        number_of_payments=1,
                        payment_option_id="",
                        requires_spending_changes=bool(
                            spending_changes
                        ),
                        completes_by_deadline=(
                            deadline is None
                            or request_date <= deadline
                        ),
                        reason=(
                            "Full payment is safely "
                            "affordable on the request date."
                        ),
                    )
                )

        # --------------------------------------------------------------
        # PARTIAL PAYMENT
        # --------------------------------------------------------------

        if (
            safe_today > EPSILON
            and safe_today + EPSILON < requested_amount
            and earliest_full_date is not None
            and self._method_allowed(
                context,
                "partial_payment",
            )
            and self._allows_partial(request)
        ):
            if (
                deadline is None
                or earliest_full_date <= deadline
            ):
                partial_plan = (
                    self.payment_plan_engine
                    .build_partial_payment_plan(
                        request=request,
                        safe_amount=safe_today,
                        full_payment_date=earliest_full_date,
                    )
                )

                if (
                    partial_plan.valid
                    and self._plan_safe(
                        partial_plan,
                        forecast,
                    )
                ):
                    candidates.append(
                        DecisionCandidate(
                            affordability_status=(
                                "affordable_with_plan"
                            ),
                            payment_method=(
                                "partial_payment"
                            ),
                            amount_safe_to_pay=(
                                self._money(
                                    safe_today
                                )
                            ),
                            payment_plan=(
                                self.payment_plan_engine
                                .format_plan(
                                    partial_plan
                                )
                            ),
                            earliest_date_for_full_payment=(
                                earliest_full_date
                            ),
                            spending_changes=(
                                spending_changes
                            ),
                            payment_plan_result=(
                                partial_plan
                            ),
                            total_cost=requested_amount,
                            first_payment_date=request_date,
                            number_of_payments=2,
                            payment_option_id="",
                            requires_spending_changes=bool(
                                spending_changes
                            ),
                            completes_by_deadline=(
                                deadline is None
                                or earliest_full_date
                                <= deadline
                            ),
                            reason=(
                                "A safe partial payment can be "
                                "made now and completed on the "
                                "earliest safe full-payment date."
                            ),
                        )
                    )

        # --------------------------------------------------------------
        # INSTALLMENTS
        # --------------------------------------------------------------

        if self._method_allowed(
            context,
            "installments",
        ):
            installment_plans = (
                self.payment_plan_engine.evaluate(
                    request=request,
                    context=context,
                    payment_options=payment_options,
                    forecast=forecast,
                    safe_amount=safe_today,
                    earliest_full_payment=(
                        earliest_full_date
                    ),
                )
            )

            for plan in installment_plans:
                if (
                    not plan.valid
                    or plan.payment_method
                    != "installments"
                ):
                    continue

                if not plan.payments:
                    continue

                first_payment = (
                    plan.payments[0].date
                )

                last_payment = (
                    plan.payments[-1].date
                )

                completes = (
                    deadline is None
                    or last_payment <= deadline
                )

                if not completes:
                    continue

                candidates.append(
                    DecisionCandidate(
                        affordability_status=(
                            "affordable_with_plan"
                        ),
                        payment_method="installments",
                        amount_safe_to_pay=(
                            min(
                                requested_amount,
                                safe_today,
                            )
                        ),
                        payment_plan=(
                            self.payment_plan_engine
                            .format_plan(plan)
                        ),
                        earliest_date_for_full_payment=(
                            last_payment
                        ),
                        spending_changes=(
                            spending_changes
                        ),
                        payment_plan_result=plan,
                        total_cost=(
                            plan.total_payable
                        ),
                        first_payment_date=(
                            first_payment
                        ),
                        number_of_payments=len(
                            plan.payments
                        ),
                        payment_option_id=(
                            plan.payment_option_id
                            or ""
                        ),
                        requires_spending_changes=bool(
                            spending_changes
                        ),
                        completes_by_deadline=True,
                        reason=(
                            "A supplied installment option "
                            "safely completes by the deadline."
                        ),
                    )
                )

        return candidates

    # ==================================================================
    # RANKING
    # ==================================================================

    def _select_best_candidate(
        self,
        candidates: list[DecisionCandidate],
        deadline: Optional[date],
    ) -> DecisionCandidate:
        """
        Rank candidates according to the challenge policy.

        Priority:

          1. Must complete by deadline.
          2. Prefer no spending changes.
          3. Prefer lower total cost.
          4. Prefer earlier start.
          5. Prefer fewer payments.
          6. Prefer deterministic payment_option_id.
        """

        def ranking_key(
            candidate: DecisionCandidate,
        ):
            # Earlier completion is useful, but deadline eligibility
            # itself has already been enforced.
            last_payment = (
                candidate.payment_plan_result.last_payment_date
                if candidate.payment_plan_result
                else date.max
            )

            first_payment = (
                candidate.first_payment_date
                or date.max
            )

            # Full payment now gets a natural advantage from:
            #   no financing fee
            #   one payment
            #   earliest start
            #
            # Partial payment is still preferred over financing when
            # its total cost is equal and it starts immediately.

            method_priority = {
                "full_payment": 0,
                "partial_payment": 1,
                "installments": 2,
            }

            return (
                0
                if candidate.completes_by_deadline
                else 1,

                # No spending changes.
                0
                if not candidate.requires_spending_changes
                else 1,

                # Lower total payable cost.
                round(
                    candidate.total_cost,
                    2,
                ),

                # Earlier first payment.
                first_payment,

                # Earlier completion.
                last_payment,

                # Fewer payments.
                candidate.number_of_payments,

                # Prefer straightforward payment methods.
                method_priority.get(
                    candidate.payment_method,
                    99,
                ),

                # Stable tie breaker.
                candidate.payment_option_id
                or "",
            )

        return min(
            candidates,
            key=ranking_key,
        )

    # ==================================================================
    # AFFORDABILITY ENGINE ADAPTER
    # ==================================================================

    def _calculate_affordability(
        self,
        request: FinancialRequest,
        context: UserContext,
        events: list[FinancialEvent],
        forecast=None,
    ):
        """
        Call the affordability engine while remaining compatible with
        the implementation created earlier.
        """

        method = getattr(
            self.affordability_engine,
            "evaluate",
            None,
        )

        if not callable(method):
            raise AttributeError(
                "AffordabilityEngine must expose evaluate()."
            )

        attempts = [
            lambda: method(
                request=request,
                context=context,
                events=events,
                forecast=forecast,
            ),
            lambda: method(
                request,
                context,
                events,
                forecast,
            ),
            lambda: method(
                request=request,
                user_context=context,
                events=events,
                forecast=forecast,
            ),
        ]

        last_error = None

        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
                continue

        raise last_error

    def _reforecast(
        self,
        request: FinancialRequest,
        context: UserContext,
        events: list[FinancialEvent],
    ):
        """
        Rebuild the forecast after permitted spending changes.

        If the affordability engine/forecaster does not expose a
        forecasting API, return None and let its evaluate() implementation
        handle the supplied events.
        """

        engine = self.affordability_engine

        for name in (
            "forecast",
            "build_forecast",
            "project",
        ):
            method = getattr(
                engine,
                name,
                None,
            )

            if not callable(method):
                continue

            attempts = [
                lambda m=method: m(
                    context=context,
                    events=events,
                    request=request,
                ),
                lambda m=method: m(
                    context=context,
                    events=events,
                ),
                lambda m=method: m(
                    request=request,
                    events=events,
                ),
                lambda m=method: m(events),
            ]

            for attempt in attempts:
                try:
                    return attempt()
                except TypeError:
                    continue
                except Exception:
                    return None

        return None

    # ==================================================================
    # AFFORDABILITY RESULT EXTRACTION
    # ==================================================================

    def _extract_safe_amount(
        self,
        affordability,
    ) -> float:
        if affordability is None:
            return 0.0

        for name in (
            "amount_safe_to_pay",
            "safe_amount",
            "safe_today",
            "maximum_safe_amount",
        ):
            value = getattr(
                affordability,
                name,
                None,
            )

            if value is not None:
                return max(
                    0.0,
                    self._money(value),
                )

        if isinstance(
            affordability,
            dict,
        ):
            for name in (
                "amount_safe_to_pay",
                "safe_amount",
                "safe_today",
                "maximum_safe_amount",
            ):
                if name in affordability:
                    return max(
                        0.0,
                        self._money(
                            affordability[name]
                        ),
                    )

        return 0.0

    def _extract_earliest_full_payment_date(
        self,
        affordability,
    ) -> Optional[date]:
        if affordability is None:
            return None

        for name in (
            "earliest_date_for_full_payment",
            "earliest_full_payment_date",
            "earliest_full_date",
        ):
            value = getattr(
                affordability,
                name,
                None,
            )

            parsed = self._date_only(value)

            if parsed is not None:
                return parsed

        if isinstance(
            affordability,
            dict,
        ):
            for name in (
                "earliest_date_for_full_payment",
                "earliest_full_payment_date",
                "earliest_full_date",
            ):
                if name in affordability:
                    parsed = self._date_only(
                        affordability[name]
                    )

                    if parsed is not None:
                        return parsed

        return None

    # ==================================================================
    # SAFETY / PERMISSIONS
    # ==================================================================

    def _plan_safe(
        self,
        plan: PaymentPlanResult,
        forecast,
    ) -> bool:
        if not plan.valid:
            return False

        if forecast is None:
            return True

        method = getattr(
            self.payment_plan_engine,
            "_schedule_is_safe",
            None,
        )

        if callable(method):
            try:
                return bool(
                    method(
                        plan,
                        forecast,
                    )
                )
            except Exception:
                return True

        return True

    def _method_allowed(
        self,
        context: UserContext,
        method: str,
    ) -> bool:
        methods = getattr(
            context,
            "payment_methods_user_will_consider",
            [],
        )

        if isinstance(methods, str):
            methods = methods.split("|")

        normalized = {
            self._normalize_method(item)
            for item in methods
        }

        return (
            self._normalize_method(method)
            in normalized
        )

    # ==================================================================
    # FINAL CONVERSION
    # ==================================================================

    def _candidate_to_decision(
        self,
        candidate: DecisionCandidate,
    ) -> Decision:
        return Decision(
            amount_safe_to_pay=self._money(
                candidate.amount_safe_to_pay
            ),
            affordability_status=(
                candidate.affordability_status
            ),
            recommended_payment_method=(
                candidate.payment_method
            ),
            payment_plan=(
                candidate.payment_plan
            ),
            earliest_date_for_full_payment=(
                candidate.earliest_date_for_full_payment
                .isoformat()
                if candidate.earliest_date_for_full_payment
                else ""
            ),
            spending_changes_needed=(
                self._format_spending_changes(
                    candidate.spending_changes
                )
            ),
            payment_option_id=(
                candidate.payment_option_id
            ),
            total_cost=self._money(
                candidate.total_cost
            ),
            reason=candidate.reason,
        )

    @staticmethod
    def _format_spending_changes(
        changes: Iterable[SpendingChange],
    ) -> str:
        changes = list(changes)

        if not changes:
            return "none"

        return "|".join(
            change.output()
            for change in changes
        )

    # ==================================================================
    # GENERIC HELPERS
    # ==================================================================

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

    @staticmethod
    def _money(value) -> float:
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
        }

        return aliases.get(
            value,
            value,
        )

    @staticmethod
    def _date_only(value) -> Optional[date]:
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