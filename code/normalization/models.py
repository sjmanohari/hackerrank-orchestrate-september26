"""
Typed domain models for the Buy or Wait? challenge.

These models normalize raw CSV records into predictable Python objects.
They intentionally do not contain decision-making logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================================
# Helpers
# ============================================================================

def _clean_string(
    value: Any,
    default: str = "",
) -> str:
    """
    Convert a value to a clean string.
    """

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    return text if text else default


def _optional_string(
    value: Any,
) -> Optional[str]:
    """
    Convert a value to Optional[str].
    """

    value = _clean_string(value)

    return value if value else None


def _float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Safely convert a value to float.
    """

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(
    value: Any,
) -> Optional[float]:
    """
    Safely convert a value to Optional[float].

    Blank values remain None. This is important for financial events because
    a missing amount is NOT equivalent to zero.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, str) and not value.strip():
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _date(
    value: Any,
) -> Optional[date]:
    """
    Parse a date-like value.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, date):
        return value

    parsed = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(parsed):
        return None

    return parsed.date()


def _bool(
    value: Any,
    default: bool = False,
) -> bool:
    """
    Parse common CSV boolean representations.
    """

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {
        "true",
        "1",
        "yes",
        "y",
        "on",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
        "n",
        "off",
    }:
        return False

    return default


def _extra_columns(
    row: pd.Series,
    known_columns: set[str],
) -> Dict[str, Any]:
    """
    Preserve unknown CSV columns without letting them affect decisions.
    """

    extra: Dict[str, Any] = {}

    for column, value in row.items():
        if column in known_columns:
            continue

        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass

        extra[str(column)] = value

    return extra


# ============================================================================
# User profile
# ============================================================================

@dataclass
class UserContext:
    user_id: str
    home_currency: str
    available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: str = ""
    spending_preferences: str = ""
    payment_methods_user_will_consider: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_series(
        cls,
        row: pd.Series,
    ) -> "UserContext":
        known = {
            "user_id",
            "home_currency",
            "available_balance",
            "minimum_balance_to_keep",
            "financial_priorities",
            "spending_preferences",
            "payment_methods_user_will_consider",
        }

        return cls(
            user_id=_clean_string(row.get("user_id")),
            home_currency=_clean_string(
                row.get("home_currency"),
                "INR",
            ).upper(),
            available_balance=_float(
                row.get("available_balance")
            ),
            minimum_balance_to_keep=_float(
                row.get("minimum_balance_to_keep")
            ),
            financial_priorities=_clean_string(
                row.get("financial_priorities")
            ),
            spending_preferences=_clean_string(
                row.get("spending_preferences")
            ),
            payment_methods_user_will_consider=_clean_string(
                row.get("payment_methods_user_will_consider")
            ),
            extra=_extra_columns(row, known),
        )


# ============================================================================
# Financial event
# ============================================================================

@dataclass
class FinancialEvent:
    event_id: str
    user_id: str
    event_date: Optional[date]
    event_type: str
    amount: Optional[float]
    currency: str
    status: str
    linked_event_id: Optional[str] = None
    recurring: bool = False
    flexible: bool = False
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def amount_resolved(self) -> bool:
        """
        Whether the event has a known amount.
        """

        return self.amount is not None

    @classmethod
    def from_series(
        cls,
        row: pd.Series,
    ) -> "FinancialEvent":
        known = {
            "event_id",
            "user_id",
            "event_date",
            "date",
            "event_type",
            "type",
            "amount",
            "currency",
            "status",
            "linked_event_id",
            "recurring",
            "flexible",
            "description",
        }

        event_date = row.get("event_date")

        if event_date is None:
            event_date = row.get("date")

        event_type = row.get("event_type")

        if event_type is None:
            event_type = row.get("type")

        return cls(
            event_id=_clean_string(row.get("event_id")),
            user_id=_clean_string(row.get("user_id")),
            event_date=_date(event_date),
            event_type=_clean_string(event_type).lower(),
            amount=_optional_float(row.get("amount")),
            currency=_clean_string(
                row.get("currency"),
                "INR",
            ).upper(),
            status=_clean_string(
                row.get("status"),
                "confirmed",
            ).lower(),
            linked_event_id=_optional_string(
                row.get("linked_event_id")
            ),
            recurring=_bool(
                row.get("recurring")
            ),
            flexible=_bool(
                row.get("flexible")
            ),
            description=_clean_string(
                row.get("description")
            ),
            extra=_extra_columns(row, known),
        )


# ============================================================================
# Financial request
# ============================================================================

@dataclass
class FinancialRequest:
    request_id: str
    user_id: str
    request_date: Optional[date]
    request_type: str
    requested_amount: float
    desired_completion_date: Optional[date]
    allows_partial_payment: bool
    request_text: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_series(
        cls,
        row: pd.Series,
    ) -> "FinancialRequest":
        known = {
            "request_id",
            "user_id",
            "request_date",
            "date",
            "request_type",
            "type",
            "requested_amount",
            "amount",
            "desired_completion_date",
            "completion_date",
            "deadline",
            "allows_partial_payment",
            "partial_payment",
            "request_text",
            "text",
        }

        request_date = row.get("request_date")

        if request_date is None:
            request_date = row.get("date")

        request_type = row.get("request_type")

        if request_type is None:
            request_type = row.get("type")

        requested_amount = row.get("requested_amount")

        if requested_amount is None:
            requested_amount = row.get("amount")

        completion_date = row.get(
            "desired_completion_date"
        )

        if completion_date is None:
            completion_date = row.get("completion_date")

        if completion_date is None:
            completion_date = row.get("deadline")

        partial = row.get("allows_partial_payment")

        if partial is None:
            partial = row.get("partial_payment")

        request_text = row.get("request_text")

        if request_text is None:
            request_text = row.get("text")

        return cls(
            request_id=_clean_string(
                row.get("request_id")
            ),
            user_id=_clean_string(
                row.get("user_id")
            ),
            request_date=_date(request_date),
            request_type=_clean_string(
                request_type
            ).lower(),
            requested_amount=max(
                0.0,
                _float(requested_amount),
            ),
            desired_completion_date=_date(
                completion_date
            ),
            allows_partial_payment=_bool(
                partial
            ),
            request_text=_clean_string(
                request_text
            ),
            extra=_extra_columns(row, known),
        )


# ============================================================================
# Payment option
# ============================================================================

@dataclass
class PaymentOption:
    payment_option_id: str
    request_id: str
    starts_on: Optional[date]
    days_between_payments: int
    number_of_payments: int
    financing_fee: float
    total_payable: float
    currency: str
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_series(
        cls,
        row: pd.Series,
    ) -> "PaymentOption":
        known = {
            "payment_option_id",
            "request_id",
            "starts_on",
            "start_date",
            "days_between_payments",
            "number_of_payments",
            "financing_fee",
            "total_payable",
            "currency",
        }

        starts_on = row.get("starts_on")

        if starts_on is None:
            starts_on = row.get("start_date")

        return cls(
            payment_option_id=_clean_string(
                row.get("payment_option_id")
            ),
            request_id=_clean_string(
                row.get("request_id")
            ),
            starts_on=_date(starts_on),
            days_between_payments=max(
                0,
                int(
                    _float(
                        row.get(
                            "days_between_payments"
                        )
                    )
                ),
            ),
            number_of_payments=max(
                1,
                int(
                    _float(
                        row.get(
                            "number_of_payments"
                        ),
                        1,
                    )
                ),
            ),
            financing_fee=max(
                0.0,
                _float(
                    row.get("financing_fee")
                ),
            ),
            total_payable=max(
                0.0,
                _float(
                    row.get("total_payable")
                ),
            ),
            currency=_clean_string(
                row.get("currency"),
                "INR",
            ).upper(),
            extra=_extra_columns(row, known),
        )

    def payment_dates(
        self,
    ) -> List[date]:
        """
        Generate chronological installment dates.
        """

        if self.starts_on is None:
            return []

        from datetime import timedelta

        return [
            self.starts_on
            + timedelta(
                days=self.days_between_payments * index
            )
            for index in range(self.number_of_payments)
        ]


# ============================================================================
# Message
# ============================================================================

@dataclass
class Message:
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    message_text: str
    message_date: Optional[date]
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_series(
        cls,
        row: pd.Series,
    ) -> "Message":
        known = {
            "message_id",
            "user_id",
            "request_id",
            "related_request_id",
            "related_event_id",
            "message",
            "message_text",
            "text",
            "content",
            "body",
            "message_date",
            "date",
        }

        request_id = row.get("request_id")

        if request_id is None:
            request_id = row.get("related_request_id")

        text = row.get("message_text")

        if text is None:
            text = row.get("message")

        if text is None:
            text = row.get("text")

        if text is None:
            text = row.get("content")

        if text is None:
            text = row.get("body")

        message_date = row.get("message_date")

        if message_date is None:
            message_date = row.get("date")

        return cls(
            message_id=_clean_string(
                row.get("message_id")
            ),
            user_id=_clean_string(
                row.get("user_id")
            ),
            request_id=_optional_string(
                request_id
            ),
            related_event_id=_optional_string(
                row.get("related_event_id")
            ),
            message_text=_clean_string(text),
            message_date=_date(message_date),
            extra=_extra_columns(row, known),
        )


# ============================================================================
# Image reference
# ============================================================================

@dataclass
class ImageReference:
    image_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    image_path: Optional[str]
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_series(
        cls,
        row: pd.Series,
    ) -> "ImageReference":
        known = {
            "image_id",
            "user_id",
            "request_id",
            "related_request_id",
            "related_event_id",
            "image_path",
            "path",
            "image",
            "description",
        }

        request_id = row.get("request_id")

        if request_id is None:
            request_id = row.get(
                "related_request_id"
            )

        image_path = row.get("image_path")

        if image_path is None:
            image_path = row.get("path")

        if image_path is None:
            image_path = row.get("image")

        return cls(
            image_id=_clean_string(
                row.get("image_id")
            ),
            user_id=_clean_string(
                row.get("user_id")
            ),
            request_id=_optional_string(
                request_id
            ),
            related_event_id=_optional_string(
                row.get("related_event_id")
            ),
            image_path=_optional_string(
                image_path
            ),
            description=_clean_string(
                row.get("description")
            ),
            extra=_extra_columns(row, known),
        )