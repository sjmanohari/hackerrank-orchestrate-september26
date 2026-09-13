from __future__ import annotations

from datetime import date
from typing import Iterable


class CurrencyConverter:
    """
    Converts amounts using ONLY the supplied dated exchange rates.

    No live FX lookup and no invented fallback rate is allowed.
    """

    def __init__(self, exchange_rates: Iterable):
        self._rates: dict[tuple[date, str, str], float] = {}

        for row in exchange_rates:
            rate_date = self._get(row, "rate_date")
            from_currency = self._get(row, "from_currency")
            to_currency = self._get(row, "to_currency")
            rate = self._get(row, "rate")

            if not rate_date or not from_currency or not to_currency:
                continue

            try:
                parsed_date = (
                    rate_date
                    if isinstance(rate_date, date)
                    else date.fromisoformat(str(rate_date))
                )

                self._rates[
                    (
                        parsed_date,
                        str(from_currency).upper(),
                        str(to_currency).upper(),
                    )
                ] = float(rate)

            except (TypeError, ValueError):
                continue

    def convert(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        rate_date: date,
    ) -> float | None:
        """
        Return converted amount or None when the supplied dataset
        does not contain the required dated conversion.
        """

        source = str(from_currency).upper()
        target = str(to_currency).upper()

        if source == target:
            return float(amount)

        key = (rate_date, source, target)

        if key in self._rates:
            return float(amount) * self._rates[key]

        reverse_key = (rate_date, target, source)

        if reverse_key in self._rates:
            reverse_rate = self._rates[reverse_key]

            if reverse_rate == 0:
                return None

            return float(amount) / reverse_rate

        return None

    @staticmethod
    def _get(row, name: str):
        if isinstance(row, dict):
            return row.get(name)

        return getattr(row, name, None)