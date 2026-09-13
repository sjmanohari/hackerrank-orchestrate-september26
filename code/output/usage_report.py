"""
Model usage and cost reporting.

The hackathon submission requires an evaluation/usage_report.md describing:

- model providers
- model names
- number of model calls
- input tokens
- output tokens
- total tokens
- average tokens/request
- estimated cost

This module records only usage actually reported by the model adapters.

No token count or price is fabricated when the provider does not return it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class ModelUsage:
    """
    Aggregated usage for one model/provider pair.
    """

    provider: str
    model: str

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    # None means that a reliable provider-specific price was not configured.
    estimated_cost_usd: Optional[float] = None

    def add(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        estimated_cost_usd: Optional[float] = None,
    ) -> None:
        self.calls += 1

        if input_tokens is not None:
            self.input_tokens += max(
                0,
                int(input_tokens),
            )

        if output_tokens is not None:
            self.output_tokens += max(
                0,
                int(output_tokens),
            )

        if estimated_cost_usd is not None:
            if self.estimated_cost_usd is None:
                self.estimated_cost_usd = 0.0

            self.estimated_cost_usd += float(
                estimated_cost_usd
            )

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
        )


@dataclass
class UsageReporter:
    """
    Collects model usage during one complete run.
    """

    report_path: str | Path = (
        "evaluation/usage_report.md"
    )

    # Optional configured prices:
    # provider/model -> input USD per 1M tokens
    input_price_per_million: Dict[str, float] = field(
        default_factory=dict
    )

    # provider/model -> output USD per 1M tokens
    output_price_per_million: Dict[str, float] = field(
        default_factory=dict
    )

    usage: Dict[str, ModelUsage] = field(
        default_factory=dict
    )

    requests_processed: int = 0

    def register_model(
        self,
        provider: str,
        model: str,
    ) -> None:
        """
        Register a provider/model even if no model call happens.
        """

        key = self._key(
            provider,
            model,
        )

        if key not in self.usage:
            self.usage[key] = ModelUsage(
                provider=str(provider),
                model=str(model),
            )

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        estimated_cost_usd: Optional[float] = None,
    ) -> None:
        """
        Record one model call.

        If an explicit cost is supplied, use it.

        Otherwise, if token pricing has been configured for the
        provider/model pair, calculate the estimated cost.
        """

        key = self._key(
            provider,
            model,
        )

        if key not in self.usage:
            self.usage[key] = ModelUsage(
                provider=str(provider),
                model=str(model),
            )

        if estimated_cost_usd is None:
            estimated_cost_usd = self._estimate_cost(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        self.usage[key].add(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

    def record_request(self) -> None:
        """
        Increment the number of processed requests.
        """

        self.requests_processed += 1

    def record_requests(
        self,
        count: int,
    ) -> None:
        self.requests_processed += max(
            0,
            int(count),
        )

    def total_calls(self) -> int:
        return sum(
            item.calls
            for item in self.usage.values()
        )

    def total_input_tokens(self) -> int:
        return sum(
            item.input_tokens
            for item in self.usage.values()
        )

    def total_output_tokens(self) -> int:
        return sum(
            item.output_tokens
            for item in self.usage.values()
        )

    def total_tokens(self) -> int:
        return (
            self.total_input_tokens()
            + self.total_output_tokens()
        )

    def total_estimated_cost_usd(self) -> Optional[float]:
        """
        Return total cost if every recorded cost is known.

        If any model has unknown pricing, return None rather than
        presenting a misleading total.
        """

        if not self.usage:
            return 0.0

        costs = [
            item.estimated_cost_usd
            for item in self.usage.values()
        ]

        if any(
            cost is None
            for cost in costs
        ):
            return None

        return sum(
            float(cost)
            for cost in costs
            if cost is not None
        )

    def average_tokens_per_request(
        self,
    ) -> Optional[float]:
        if self.requests_processed <= 0:
            return None

        return (
            self.total_tokens()
            / self.requests_processed
        )

    def generate_markdown(self) -> str:
        """
        Generate the complete usage_report.md contents.
        """

        generated_at = datetime.now(
            timezone.utc
        ).isoformat()

        total_cost = (
            self.total_estimated_cost_usd()
        )

        average_tokens = (
            self.average_tokens_per_request()
        )

        lines: List[str] = [
            "# Model Usage Report",
            "",
            "Generated automatically by "
            "`code/output/usage_report.py`.",
            "",
            f"- Generated at (UTC): `{generated_at}`",
            f"- Requests processed: `{self.requests_processed}`",
            f"- Total model calls: `{self.total_calls()}`",
            f"- Total input tokens: `{self.total_input_tokens()}`",
            f"- Total output tokens: `{self.total_output_tokens()}`",
            f"- Total tokens: `{self.total_tokens()}`",
        ]

        if average_tokens is None:
            lines.append(
                "- Average tokens/request: `N/A`"
            )
        else:
            lines.append(
                "- Average tokens/request: "
                f"`{average_tokens:.2f}`"
            )

        if total_cost is None:
            lines.append(
                "- Estimated total cost (USD): `N/A` "
                "(pricing not configured for every model)"
            )
        else:
            lines.append(
                "- Estimated total cost (USD): "
                f"`${total_cost:.6f}`"
            )

        lines.extend(
            [
                "",
                "## Models",
                "",
                "| Provider | Model | Calls | Input tokens | "
                "Output tokens | Total tokens | Estimated cost (USD) |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )

        for item in sorted(
            self.usage.values(),
            key=lambda value: (
                value.provider,
                value.model,
            ),
        ):
            cost = (
                "N/A"
                if item.estimated_cost_usd is None
                else f"{item.estimated_cost_usd:.6f}"
            )

            lines.append(
                "| "
                f"{self._escape(item.provider)} | "
                f"{self._escape(item.model)} | "
                f"{item.calls} | "
                f"{item.input_tokens} | "
                f"{item.output_tokens} | "
                f"{item.total_tokens} | "
                f"{cost} |"
            )

        lines.extend(
            [
                "",
                "## Notes",
                "",
                "- Token counts are recorded only when returned by "
                "the configured model provider.",
                "- Costs are estimates, not billing statements.",
                "- If pricing is not configured for a model, its cost "
                "is reported as `N/A`.",
                "- Deterministic financial calculations do not count "
                "as model calls.",
                "",
            ]
        )

        return "\n".join(lines)

    def write(
        self,
        report_path: Optional[str | Path] = None,
    ) -> Path:
        """
        Write evaluation/usage_report.md.
        """

        path = Path(
            report_path
            if report_path is not None
            else self.report_path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            self.generate_markdown(),
            encoding="utf-8",
        )

        return path

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def configure_price(
        self,
        provider: str,
        model: str,
        input_usd_per_million_tokens: Optional[float] = None,
        output_usd_per_million_tokens: Optional[float] = None,
    ) -> None:
        key = self._key(
            provider,
            model,
        )

        if input_usd_per_million_tokens is not None:
            self.input_price_per_million[key] = float(
                input_usd_per_million_tokens
            )

        if output_usd_per_million_tokens is not None:
            self.output_price_per_million[key] = float(
                output_usd_per_million_tokens
            )

    def _estimate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> Optional[float]:
        key = self._key(
            provider,
            model,
        )

        input_price = self.input_price_per_million.get(
            key
        )

        output_price = self.output_price_per_million.get(
            key
        )

        # We need pricing for every token category that has usage.
        if input_tokens and input_price is None:
            return None

        if output_tokens and output_price is None:
            return None

        cost = 0.0

        if input_tokens and input_price is not None:
            cost += (
                input_tokens
                / 1_000_000
                * input_price
            )

        if output_tokens and output_price is not None:
            cost += (
                output_tokens
                / 1_000_000
                * output_price
            )

        return cost

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(
        provider: str,
        model: str,
    ) -> str:
        return (
            f"{str(provider).strip().lower()}"
            f"/"
            f"{str(model).strip()}"
        )

    @staticmethod
    def _escape(value: str) -> str:
        return (
            str(value)
            .replace("|", "\\|")
            .replace("\n", " ")
            .replace("\r", " ")
        )