from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterable


class ConflictResolver:
    """
    Reconstructs the most financially defensible event state.

    Priority:
      1. explicit cancellation / settlement / amendment
      2. newer record from same source
      3. settled event over estimate
      4. safer interpretation when unresolved
    """

    TERMINAL_STATUSES = {
        "cancelled",
        "failed",
        "settled",
    }

    def resolve_events(
        self,
        *,
        events: Iterable,
        messages: Iterable | None = None,
        images: Iterable | None = None,
    ) -> list:
        events = list(events)

        messages = list(messages or [])
        images = list(images or [])

        messages_by_event = defaultdict(list)

        for message in messages:
            related = self._get(
                message,
                "related_event_id",
            )

            if related:
                messages_by_event[related].append(
                    message
                )

        images_by_event = defaultdict(list)

        for image in images:
            related = self._get(
                image,
                "related_event_id",
            )

            if related:
                images_by_event[related].append(
                    image
                )

        # First remove explicitly cancelled/invalidated events.
        cancelled_ids = set()

        for event in events:
            status = str(
                self._get(event, "status") or ""
            ).lower()

            if status == "cancelled":
                cancelled_ids.add(
                    self._get(event, "event_id")
                )

            linked = self._get(
                event,
                "linked_event_id",
            )

            if linked and status == "cancelled":
                cancelled_ids.add(linked)

        result = []

        for event in events:
            event_id = self._get(
                event,
                "event_id",
            )

            status = str(
                self._get(event, "status") or ""
            ).lower()

            if event_id in cancelled_ids:
                continue

            # Attach supporting information without trusting its
            # instructions as executable commands.
            try:
                setattr(
                    event,
                    "_related_messages",
                    messages_by_event.get(
                        event_id,
                        [],
                    ),
                )

                setattr(
                    event,
                    "_related_images",
                    images_by_event.get(
                        event_id,
                        [],
                    ),
                )
            except Exception:
                pass

            result.append(event)

        return self._deduplicate(result)

    def _deduplicate(self, events: list) -> list:
        """
        Prefer the strongest lifecycle state for linked events.
        """

        grouped = defaultdict(list)

        for event in events:
            event_id = self._get(
                event,
                "event_id",
            )

            linked_id = self._get(
                event,
                "linked_event_id",
            )

            key = linked_id or event_id

            grouped[key].append(event)

        selected = []

        for group in grouped.values():
            best = max(
                group,
                key=self._priority,
            )

            selected.append(best)

        selected.sort(
            key=lambda event: (
                self._date_value(
                    self._get(event, "event_date")
                ),
                str(
                    self._get(event, "event_id")
                    or ""
                ),
            )
        )

        return selected

    def _priority(self, event) -> tuple:
        status = str(
            self._get(event, "status") or ""
        ).lower()

        return (
            1 if status == "settled" else 0,
            1 if status == "scheduled" else 0,
            0 if status == "pending" else -1,
            self._datetime_value(
                self._get(event, "settlement_date")
            ),
            self._datetime_value(
                self._get(event, "event_date")
            ),
        )

    @staticmethod
    def _get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)

        return getattr(obj, key, None)

    @staticmethod
    def _date_value(value):
        if not value:
            return ""

        return str(value)

    @staticmethod
    def _datetime_value(value):
        if not value:
            return 0.0

        try:
            return datetime.fromisoformat(
                str(value)
            ).timestamp()
        except (TypeError, ValueError):
            return 0.0