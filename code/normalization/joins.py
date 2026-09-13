from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import (
    FinancialEvent,
    FinancialRequest,
    ImageReference,
    Message,
    PaymentOption,
    UserContext,
)


class DatasetJoiner:
    """
    Provides normalized access to all dataset entities and the
    relationships between them.

    Expected datasets:
        financial_profiles.csv
        financial_events.csv
        requests.csv
        request_payment_options.csv
        messages.csv
        images.csv
    """

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)

        self.profiles: list[Any] = []
        self.events: list[Any] = []
        self.requests: list[Any] = []
        self.payment_options: list[Any] = []
        self.messages: list[Any] = []
        self.images: list[Any] = []

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_all(self):
        """
        Load all supported CSV files.

        This method expects the project's DatasetLoader to exist.
        """

        from ..ingestion.dataset_loader import DatasetLoader

        loader = DatasetLoader(self.dataset_root)
        dataset = loader.load()

        self.profiles = self._get_collection(
            dataset,
            "profiles",
            "financial_profiles",
        )

        self.events = self._get_collection(
            dataset,
            "events",
            "financial_events",
        )

        self.requests = self._get_collection(
            dataset,
            "requests",
        )

        self.payment_options = self._get_collection(
            dataset,
            "payment_options",
            "request_payment_options",
        )

        self.messages = self._get_collection(
            dataset,
            "messages",
        )

        self.images = self._get_collection(
            dataset,
            "images",
        )

        return dataset

    # ------------------------------------------------------------------
    # User context
    # ------------------------------------------------------------------

    def build_user_context(
        self,
        user_id: str,
    ) -> UserContext:
        """
        Convert the user's financial profile into UserContext.

        The actual financial_profiles.csv schema has separate fields
        for categories the user is willing to reduce and categories
        the user is willing to stop.
        """

        profile = self._find_one(
            self.profiles,
            "user_id",
            user_id,
        )

        if profile is None:
            raise ValueError(
                f"No financial profile found for {user_id}"
            )

        return self._profile_to_context(profile)

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    def request_by_id(
        self,
        request_id: str,
    ) -> FinancialRequest | None:

        row = self._find_one(
            self.requests,
            "request_id",
            request_id,
        )

        if row is None:
            return None

        if isinstance(row, FinancialRequest):
            return row

        return FinancialRequest(
            request_id=str(
                self._value(row, "request_id")
            ),
            user_id=str(
                self._value(row, "user_id")
            ),
            request_date=self._date(
                self._value(row, "request_date")
            ),
            request_type=str(
                self._value(row, "request_type")
                or ""
            ),
            requested_amount=float(
                self._value(
                    row,
                    "requested_amount",
                )
                or 0
            ),
            desired_completion_date=self._date(
                self._value(
                    row,
                    "desired_completion_date",
                )
            ),
            allows_partial_payment=self._bool(
                self._value(
                    row,
                    "allows_partial_payment",
                )
            ),
            request_text=str(
                self._value(
                    row,
                    "request_text",
                )
                or ""
            ),
        )

    # ------------------------------------------------------------------
    # Financial events
    # ------------------------------------------------------------------

    def events_for_user(
        self,
        user_id: str,
    ) -> list[FinancialEvent]:

        rows = [
            row
            for row in self.events
            if self._value(
                row,
                "user_id",
            ) == user_id
        ]

        result = []

        for row in rows:
            if isinstance(row, FinancialEvent):
                result.append(row)
            else:
                result.append(
                    self._event_from_row(row)
                )

        return result

    def event_by_id(
        self,
        event_id: str,
    ) -> FinancialEvent | None:

        row = self._find_one(
            self.events,
            "event_id",
            event_id,
        )

        if row is None:
            return None

        if isinstance(row, FinancialEvent):
            return row

        return self._event_from_row(row)

    def events_for_request(
        self,
        request_id: str,
    ) -> list[FinancialEvent]:

        request = self.request_by_id(
            request_id
        )

        if request is None:
            return []

        return self.events_for_user(
            request.user_id
        )

    # ------------------------------------------------------------------
    # Payment options
    # ------------------------------------------------------------------

    def payment_options_for_request(
        self,
        request_id: str,
    ) -> list[PaymentOption]:

        rows = [
            row
            for row in self.payment_options
            if self._value(
                row,
                "request_id",
            ) == request_id
        ]

        result = []

        for row in rows:
            if isinstance(row, PaymentOption):
                result.append(row)
            else:
                result.append(
                    self._payment_option_from_row(
                        row
                    )
                )

        return result

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def messages_for_request(
        self,
        request_id: str,
    ) -> list[Message]:

        rows = [
            row
            for row in self.messages
            if self._value(
                row,
                "request_id",
            ) == request_id
        ]

        result = []

        for row in rows:
            if isinstance(row, Message):
                result.append(row)
            else:
                result.append(
                    self._message_from_row(row)
                )

        return result

    def messages_for_event(
        self,
        event_id: str,
    ) -> list[Message]:

        rows = [
            row
            for row in self.messages
            if self._value(
                row,
                "related_event_id",
            ) == event_id
        ]

        result = []

        for row in rows:
            if isinstance(row, Message):
                result.append(row)
            else:
                result.append(
                    self._message_from_row(row)
                )

        return result

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def images_for_request(
        self,
        request_id: str,
    ) -> list[ImageReference]:

        rows = [
            row
            for row in self.images
            if self._value(
                row,
                "request_id",
            ) == request_id
        ]

        return [
            row
            if isinstance(row, ImageReference)
            else self._image_from_row(row)
            for row in rows
        ]

    def images_for_event(
        self,
        event_id: str,
    ) -> list[ImageReference]:

        rows = [
            row
            for row in self.images
            if self._value(
                row,
                "related_event_id",
            ) == event_id
        ]

        return [
            row
            if isinstance(row, ImageReference)
            else self._image_from_row(row)
            for row in rows
        ]

    def image_by_id(
        self,
        image_id: str,
    ) -> ImageReference | None:

        row = self._find_one(
            self.images,
            "image_id",
            image_id,
        )

        if row is None:
            return None

        if isinstance(row, ImageReference):
            return row

        return self._image_from_row(row)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _profile_to_context(
        self,
        profile,
    ) -> UserContext:

        reduce_categories = self._split(
            self._value(
                profile,
                "expense_categories_user_is_willing_to_reduce",
            )
        )

        stop_categories = self._split(
            self._value(
                profile,
                "expense_categories_user_is_willing_to_stop",
            )
        )

        protected_categories = self._split(
            self._value(
                profile,
                "expense_categories_to_protect",
            )
        )

        payment_methods = self._split(
            self._value(
                profile,
                "payment_methods_user_will_consider",
            )
        )

        # Keep flexible_categories for compatibility with the
        # earlier forecasting/decision modules.
        flexible_categories = sorted(
            set(reduce_categories)
            | set(stop_categories)
        )

        return UserContext(
            user_id=str(
                self._value(
                    profile,
                    "user_id",
                )
            ),
            home_currency=str(
                self._value(
                    profile,
                    "home_currency",
                )
                or ""
            ),
            current_available_balance=float(
                self._value(
                    profile,
                    "current_available_balance",
                )
                or 0
            ),
            minimum_balance_to_keep=float(
                self._value(
                    profile,
                    "minimum_balance_to_keep",
                )
                or 0
            ),
            financial_priorities=self._split(
                self._value(
                    profile,
                    "financial_priorities",
                )
            ),
            protected_categories=protected_categories,
            flexible_categories=flexible_categories,
            stop_categories=stop_categories,
            payment_methods_user_will_consider=payment_methods,
            max_installment_months=self._optional_int(
                self._value(
                    profile,
                    "max_installment_months",
                )
            ),
        )

    def _event_from_row(
        self,
        row,
    ) -> FinancialEvent:

        return FinancialEvent(
            event_id=str(
                self._value(
                    row,
                    "event_id",
                )
            ),
            user_id=str(
                self._value(
                    row,
                    "user_id",
                )
            ),
            event_type=str(
                self._value(
                    row,
                    "event_type",
                )
                or ""
            ),
            description=str(
                self._value(
                    row,
                    "description",
                )
                or ""
            ),
            category=str(
                self._value(
                    row,
                    "category",
                )
                or ""
            ),
            direction=str(
                self._value(
                    row,
                    "direction",
                )
                or ""
            ),
            amount=self._optional_float(
                self._value(
                    row,
                    "amount",
                )
            ),
            currency=str(
                self._value(
                    row,
                    "currency",
                )
                or ""
            ),
            event_date=self._date(
                self._value(
                    row,
                    "event_date",
                )
            ),
            settlement_date=self._date(
                self._value(
                    row,
                    "settlement_date",
                )
            ),
            status=str(
                self._value(
                    row,
                    "status",
                )
                or ""
            ),
            linked_event_id=self._optional_str(
                self._value(
                    row,
                    "linked_event_id",
                )
            ),
            flexibility=str(
                self._value(
                    row,
                    "flexibility",
                )
                or ""
            ),
            minimum_allowed_amount=self._optional_float(
                self._value(
                    row,
                    "minimum_allowed_amount",
                )
            ),
        )

    def _payment_option_from_row(
        self,
        row,
    ) -> PaymentOption:

        return PaymentOption(
            payment_option_id=str(
                self._value(
                    row,
                    "payment_option_id",
                )
            ),
            request_id=str(
                self._value(
                    row,
                    "request_id",
                )
            ),
            payment_method=str(
                self._value(
                    row,
                    "payment_method",
                )
                or ""
            ),
            payment_amount=float(
                self._value(
                    row,
                    "payment_amount",
                )
                or 0
            ),
            number_of_payments=int(
                self._value(
                    row,
                    "number_of_payments",
                )
                or 0
            ),
            first_payment_date=self._date(
                self._value(
                    row,
                    "first_payment_date",
                )
            ),
            payment_frequency_days=self._optional_int(
                self._value(
                    row,
                    "payment_frequency_days",
                )
            ),
            financing_fee=float(
                self._value(
                    row,
                    "financing_fee",
                )
                or 0
            ),
            total_payable_amount=float(
                self._value(
                    row,
                    "total_payable_amount",
                )
                or 0
            ),
        )

    def _message_from_row(
        self,
        row,
    ) -> Message:

        return Message(
            message_id=str(
                self._value(
                    row,
                    "message_id",
                )
            ),
            user_id=str(
                self._value(
                    row,
                    "user_id",
                )
            ),
            request_id=self._optional_str(
                self._value(
                    row,
                    "request_id",
                )
            ),
            related_event_id=self._optional_str(
                self._value(
                    row,
                    "related_event_id",
                )
            ),
            sent_at=str(
                self._value(
                    row,
                    "sent_at",
                )
                or ""
            ),
            source_type=str(
                self._value(
                    row,
                    "source_type",
                )
                or ""
            ),
            message_text=str(
                self._value(
                    row,
                    "message_text",
                )
                or ""
            ),
        )

    def _image_from_row(
        self,
        row,
    ) -> ImageReference:

        image_id = str(
            self._value(
                row,
                "image_id",
            )
        )

        return ImageReference(
            image_id=image_id,
            user_id=str(
                self._value(
                    row,
                    "user_id",
                )
            ),
            request_id=self._optional_str(
                self._value(
                    row,
                    "request_id",
                )
            ),
            related_event_id=self._optional_str(
                self._value(
                    row,
                    "related_event_id",
                )
            ),
            image_path=str(
                self._image_path(image_id)
            ),
        )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_collection(
        dataset,
        *names: str,
    ) -> list[Any]:

        for name in names:
            if isinstance(dataset, dict):
                value = dataset.get(name)

                if value is not None:
                    return list(value)

            value = getattr(
                dataset,
                name,
                None,
            )

            if value is not None:
                return list(value)

        return []

    @staticmethod
    def _find_one(
        rows,
        field: str,
        expected,
    ):
        for row in rows:
            if DatasetJoiner._value(
                row,
                field,
            ) == expected:
                return row

        return None

    @staticmethod
    def _value(
        row,
        field: str,
    ):
        if isinstance(row, dict):
            return row.get(field)

        return getattr(
            row,
            field,
            None,
        )

    def _image_path(
        self,
        image_id: str,
    ) -> Path:

        return (
            self.dataset_root
            / "media"
            / "images"
            / f"{image_id}.png"
        )

    @staticmethod
    def _split(value) -> list[str]:

        if value is None:
            return []

        if isinstance(value, (list, tuple, set)):
            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        return [
            item.strip()
            for item in str(value).split("|")
            if item.strip()
        ]

    @staticmethod
    def _optional_str(value):
        if value is None:
            return None

        text = str(value).strip()

        return text if text else None

    @staticmethod
    def _optional_float(value):
        if value is None or value == "":
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_int(value):
        if value is None or value == "":
            return None

        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _bool(value) -> bool:
        if isinstance(value, bool):
            return value

        return str(value).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

    @staticmethod
    def _date(value):
        from datetime import date

        if value is None or value == "":
            return None

        if isinstance(value, date):
            return value

        text = str(value).strip()

        if not text:
            return None

        return date.fromisoformat(
            text[:10]
        )