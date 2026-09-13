"""
Top-level orchestration for the Buy-or-Wait financial agent.

Pipeline:

    dataset
      ↓
    ingestion
      ↓
    normalization / joins
      ↓
    conflict resolution + currency normalization
      ↓
    optional image amount recovery
      ↓
    cash-flow forecast
      ↓
    affordability analysis
      ↓
    decision policy
      ↓
    explanation
      ↓
    validation
      ↓
    output.csv
      ↓
    evaluation/usage_report.md
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from code.decision.decision_policy import DecisionPolicy
from code.decision.explanation import ExplanationBuilder
from code.financial.payment_plans import PaymentPlanEngine
from code.forecaster.affordability import AffordabilityEngine
from code.forecaster.cash_flow import CashFlowForecaster
from code.models.vlm import ImageAmountExtractor
from code.normalization.conflict_resolution import ConflictResolver
from code.normalization.currency import CurrencyConverter
from code.normalization.joins import DatasetJoiner
from code.output.output_writer import OutputWriter
from code.output.usage_report import UsageReporter
from code.validation.consistency_checks import ConsistencyChecker
from code.validation.output_validator import OutputValidator


class Orchestrator:
    """
    Coordinates the complete financial decision pipeline.
    """

    def __init__(
        self,
        dataset_root: str | Path = "dataset",
        output_path: str | Path = "output.csv",
        usage_report_path: str | Path = (
            "evaluation/usage_report.md"
        ),
    ):
        self.dataset_root = Path(dataset_root)
        self.output_path = Path(output_path)

        self.joiner = DatasetJoiner(
            self.dataset_root
        )

        self.conflict_resolver = ConflictResolver()

        self.vlm = ImageAmountExtractor(
            dataset_root=self.dataset_root
        )

        self.forecaster = CashFlowForecaster()
        self.affordability = AffordabilityEngine(self.forecaster)
        self.payment_plans = PaymentPlanEngine()

        self.policy = DecisionPolicy(
            payment_plan_engine=self.payment_plans
        )

        self.explanation_builder = ExplanationBuilder()

        self.validator = OutputValidator()
        self.consistency_checker = ConsistencyChecker()

        self.output_writer = OutputWriter(
            self.output_path
        )

        self.usage_reporter = UsageReporter(
            report_path=usage_report_path
        )

        self.currency_converter = None

    # ==================================================================
    # Public entry point
    # ==================================================================

    def run(self) -> Path:
        """
        Execute the complete pipeline.

        Returns:
            Path to the generated output.csv.
        """

        dataset = self.joiner.load_all()

        self._configure_usage_reporting()

        requests = self._get_collection(
            dataset,
            "requests",
        )

        if not requests:
            raise RuntimeError(
                "No requests found in dataset/requests.csv."
            )

        self.usage_reporter.record_requests(
            len(requests)
        )

        self.currency_converter = CurrencyConverter(
            self._get_collection(
                dataset,
                "exchange_rates",
            )
        )

        raw_rows: List[Dict[str, Any]] = []

        for request in requests:
            row = self._process_request(
                request=request,
                dataset=dataset,
            )

            raw_rows.append(row)

        # First validate the output schema and allowed values.
        validated_rows = self.validator.validate_rows(
            raw_rows,
            requests=requests,
        )

        # Then perform cross-field financial consistency checks.
        consistency_issues = self._run_consistency_checks(
            validated_rows,
            requests=requests,
        )

        if consistency_issues:
            raise RuntimeError(
                self._format_consistency_issues(
                    consistency_issues
                )
            )

        # Stable ordering is important for reproducibility.
        validated_rows.sort(
            key=lambda row: str(
                row["request_id"]
            )
        )

        output_path = self.output_writer.write(
            validated_rows
        )

        self.usage_reporter.write()

        return output_path

    # ==================================================================
    # Request processing
    # ==================================================================

    def _process_request(
        self,
        request: Any,
        dataset: Any,
    ) -> Dict[str, Any]:
        request_id = self._value(
            request,
            "request_id",
        )

        user_id = self._value(
            request,
            "user_id",
        )

        context = self.joiner.build_user_context(
            user_id
        )

        events = self.joiner.events_for_user(
            user_id
        )

        messages = self.joiner.messages_for_request(
            request_id
        )

        images = self.joiner.images_for_request(
            request_id
        )

        payment_options = (
            self.joiner.payment_options_for_request(
                request_id
            )
        )

        # --------------------------------------------------------------
        # Resolve conflicting/duplicated financial events.
        # --------------------------------------------------------------

        events = self.conflict_resolver.resolve(
            events
        )

        # --------------------------------------------------------------
        # Recover amounts missing from structured CSV data.
        #
        # Images are treated as evidence only. They cannot directly make
        # an affordability decision.
        # --------------------------------------------------------------

        events = self._resolve_event_amounts(
            events=events,
            request=request,
            dataset=dataset,
            context=context,
        )

        # --------------------------------------------------------------
        # Normalize foreign-currency events into the user's home currency.
        # --------------------------------------------------------------

        events = self._normalize_event_currencies(
            events=events,
            context=context,
        )

        # --------------------------------------------------------------
        # Build the baseline 90-day forecast.
        # --------------------------------------------------------------

        forecast = self.forecaster.forecast(
            events=events,
            context=context,
            request_date=self._request_date(
                request
            ),
        )

        # --------------------------------------------------------------
        # Determine baseline affordability BEFORE optional spending
        # changes.
        # --------------------------------------------------------------

        affordability = self.affordability.evaluate(
            request=request,
            context=context,
            forecast=forecast,
        )

        # --------------------------------------------------------------
        # Decision policy evaluates:
        #   1. immediate payment
        #   2. partial payment
        #   3. supplied installment plans
        #   4. spending adjustments
        #   5. waiting
        #   6. not affordable
        # --------------------------------------------------------------

        decision = self.policy.decide(
            request=request,
            context=context,
            forecast=forecast,
            affordability=affordability,
            events=events,
            payment_options=payment_options,
        )

        # --------------------------------------------------------------
        # Generate deterministic explanation.
        # --------------------------------------------------------------

        explanation = self.explanation_builder.build(
            request=request,
            context=context,
            decision=decision,
            affordability=affordability,
        )

        return self._decision_to_row(
            request=request,
            decision=decision,
            explanation=explanation,
        )

    # ==================================================================
    # Amount recovery
    # ==================================================================

    def _resolve_event_amounts(
        self,
        events: Iterable[Any],
        request: Any,
        dataset: Any,
        context: Any,
    ) -> List[Any]:
        """
        Recover missing event amounts using linked images when available.

        A missing amount is NOT interpreted as zero.
        """

        resolved = []

        for event in events:
            amount = self._value(
                event,
                "amount",
            )

            if amount is not None:
                resolved.append(event)
                continue

            event_id = self._value(
                event,
                "event_id",
            )

            image_refs = (
                self.joiner.images_for_event(
                    event_id
                )
            )

            recovered_amount = None
            recovered_currency = None

            for image in image_refs:
                image_id = self._value(
                    image,
                    "image_id",
                )

                if not image_id:
                    continue

                result = self.vlm.extract_amount(
                    image_id
                )

                if result is None:
                    continue

                if isinstance(result, dict):
                    recovered_amount = result.get(
                        "amount"
                    )
                    recovered_currency = result.get(
                        "currency"
                    )
                else:
                    recovered_amount = result

                if recovered_amount is not None:
                    break

            if recovered_amount is None:
                # Leave the amount unresolved.
                # Downstream code must treat this conservatively.
                resolved.append(event)
                continue

            changes = {
                "amount": float(
                    recovered_amount
                )
            }

            if recovered_currency:
                changes["currency"] = (
                    str(recovered_currency)
                )

            try:
                event = replace(
                    event,
                    **changes,
                )
            except TypeError:
                # If the event implementation does not support one of
                # the optional fields, retain the original event rather
                # than fabricating data.
                event = replace(
                    event,
                    amount=float(
                        recovered_amount
                    ),
                )

            resolved.append(event)

        return resolved

    # ==================================================================
    # Currency normalization
    # ==================================================================

    def _normalize_event_currencies(
        self,
        events: Iterable[Any],
        context: Any,
    ) -> List[Any]:
        if self.currency_converter is None:
            return list(events)

        home_currency = (
            self._value(
                context,
                "home_currency",
            )
            or self._value(
                context,
                "currency",
            )
        )

        if not home_currency:
            return list(events)

        normalized = []

        for event in events:
            amount = self._value(
                event,
                "amount",
            )

            currency = self._value(
                event,
                "currency",
            )

            if (
                amount is None
                or not currency
                or str(currency).upper()
                == str(home_currency).upper()
            ):
                normalized.append(event)
                continue

            effective_date = (
                self._value(
                    event,
                    "settlement_date",
                )
                or self._value(
                    event,
                    "event_date",
                )
            )

            converted = self.currency_converter.convert(
                amount=float(amount),
                from_currency=str(currency),
                to_currency=str(home_currency),
                effective_date=effective_date,
            )

            if converted is None:
                # No supported fixed dated FX rate.
                # Do not invent a live or estimated rate.
                normalized.append(event)
                continue

            try:
                event = replace(
                    event,
                    amount=float(converted),
                    currency=str(home_currency),
                )
            except TypeError:
                event = replace(
                    event,
                    amount=float(converted),
                )

            normalized.append(event)

        return normalized

    # ==================================================================
    # Output conversion
    # ==================================================================

    def _decision_to_row(
        self,
        request: Any,
        decision: Any,
        explanation: str,
    ) -> Dict[str, Any]:
        request_id = self._value(
            request,
            "request_id",
        )

        safe_amount = self._value(
            decision,
            "amount_safe_to_pay",
        )

        if safe_amount is None:
            safe_amount = self._value(
                decision,
                "safe_amount",
            )

        status = self._value(
            decision,
            "affordability_status",
        )

        method = self._value(
            decision,
            "recommended_payment_method",
        )

        payment_plan = self._format_payment_plan(
            self._value(
                decision,
                "payment_plan",
            )
        )

        earliest_date = self._value(
            decision,
            "earliest_date_for_full_payment",
        )

        if earliest_date is None:
            earliest_date = self._value(
                decision,
                "earliest_full_payment_date",
            )

        spending_changes = (
            self._format_spending_changes(
                self._value(
                    decision,
                    "spending_changes",
                )
            )
        )

        return {
            "request_id": str(
                request_id
            ),
            "amount_safe_to_pay": (
                0
                if safe_amount is None
                else float(safe_amount)
            ),
            "affordability_status": (
                status or "not_affordable"
            ),
            "recommended_payment_method": (
                method or "not_recommended"
            ),
            "payment_plan": payment_plan,
            "earliest_date_for_full_payment": (
                self._format_date(
                    earliest_date
                )
            ),
            "spending_changes_needed": (
                spending_changes
            ),
            "decision_explanation": explanation,
        }

    def _format_payment_plan(
        self,
        plan: Any,
    ) -> str:
        if not plan:
            return "none"

        if isinstance(plan, str):
            return plan.strip() or "none"

        entries = []

        for payment in plan:
            if isinstance(payment, (tuple, list)):
                if len(payment) < 2:
                    continue

                payment_date = payment[0]
                amount = payment[1]

            else:
                payment_date = (
                    self._value(
                        payment,
                        "payment_date",
                    )
                    or self._value(
                        payment,
                        "date",
                    )
                )

                amount = self._value(
                    payment,
                    "amount",
                )

            formatted_date = self._format_date(
                payment_date
            )

            if not formatted_date or amount is None:
                continue

            entries.append(
                f"{formatted_date}:"
                f"{float(amount):.2f}"
            )

        return (
            "|".join(entries)
            if entries
            else "none"
        )

    def _format_spending_changes(
        self,
        changes: Any,
    ) -> str:
        if not changes:
            return "none"

        if isinstance(changes, str):
            return changes.strip() or "none"

        entries = []

        for change in changes:
            if isinstance(change, str):
                entries.append(
                    change.strip()
                )
                continue

            output_method = getattr(
                change,
                "output",
                None,
            )

            if callable(output_method):
                value = output_method()

                if value:
                    entries.append(
                        str(value)
                    )

        return (
            "|".join(
                item
                for item in entries
                if item
            )
            if entries
            else "none"
        )

    # ==================================================================
    # Validation
    # ==================================================================

    def _run_consistency_checks(
        self,
        rows: List[Dict[str, Any]],
        requests: Iterable[Any],
    ):
        request_map = {
            str(
                self._value(
                    request,
                    "request_id",
                )
            ): request
            for request in requests
        }

        issues = []

        for row in rows:
            request_id = str(
                row["request_id"]
            )

            request = request_map.get(
                request_id
            )

            payment_options = (
                self.joiner.payment_options_for_request(
                    request_id
                )
            )

            issues.extend(
                self.consistency_checker.check_row(
                    row=row,
                    request=request,
                    payment_options=payment_options,
                )
            )

        return issues

    @staticmethod
    def _format_consistency_issues(
        issues,
    ) -> str:
        lines = [
            "Final decision consistency validation failed:"
        ]

        for issue in issues[:50]:
            lines.append(
                f"- request_id={issue.request_id}; "
                f"{issue.rule}: {issue.message}"
            )

        if len(issues) > 50:
            lines.append(
                f"- ... and {len(issues) - 50} more issue(s)"
            )

        return "\n".join(lines)

    # ==================================================================
    # Usage reporting
    # ==================================================================

    def _configure_usage_reporting(self) -> None:
        """
        Register configured model providers.

        Actual calls are recorded by model adapters when usage metadata
        is available.
        """

        import os

        vlm_provider = os.getenv(
            "VLM_PROVIDER"
        )
        vlm_model = os.getenv(
            "VLM_MODEL"
        )

        if vlm_provider and vlm_model:
            self.usage_reporter.register_model(
                vlm_provider,
                vlm_model,
            )

        llm_provider = os.getenv(
            "LLM_PROVIDER"
        )
        llm_model = os.getenv(
            "LLM_MODEL"
        )

        if llm_provider and llm_model:
            self.usage_reporter.register_model(
                llm_provider,
                llm_model,
            )

    # ==================================================================
    # Dataset helpers
    # ==================================================================

    @staticmethod
    def _get_collection(
        dataset: Any,
        name: str,
    ):
        """
        Support both object-style and dictionary-style DatasetLoader
        results.
        """

        if isinstance(dataset, dict):
            return dataset.get(
                name,
                [],
            )

        value = getattr(
            dataset,
            name,
            None,
        )

        if value is None:
            return []

        return value

    @staticmethod
    def _value(
        obj: Any,
        field: str,
        default: Any = None,
    ):
        if obj is None:
            return default

        if isinstance(obj, dict):
            return obj.get(
                field,
                default,
            )

        value = getattr(
            obj,
            field,
            default,
        )

        return value

    @classmethod
    def _request_date(
        cls,
        request: Any,
    ):
        return cls._value(
            request,
            "request_date",
        )

    @staticmethod
    def _format_date(
        value: Any,
    ) -> str:
        if value is None:
            return ""

        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass

        text = str(value).strip()

        if "T" in text:
            text = text.split(
                "T",
                1,
            )[0]

        if " " in text:
            text = text.split(
                " ",
                1,
            )[0]

        return text