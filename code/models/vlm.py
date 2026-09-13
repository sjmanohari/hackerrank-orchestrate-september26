from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ImageAmountExtractor:
    """
    Optional multimodal amount extractor.

    If no VLM is configured, returns None rather than inventing an
    amount.
    """

    def __init__(
        self,
        dataset_root: str | Path,
    ):
        self.dataset_root = Path(dataset_root)

        self.provider = os.getenv(
            "VLM_PROVIDER",
            "",
        ).strip().lower()

        self.model = os.getenv(
            "VLM_MODEL",
            "",
        ).strip()

        self.api_key = os.getenv(
            "OPENAI_API_KEY",
            "",
        ).strip()

    def extract_amount(
        self,
        image_id: str,
    ) -> dict[str, Any] | None:

        image_path = (
            self.dataset_root
            / "media"
            / "images"
            / f"{image_id}.png"
        )

        if not image_path.exists():
            return None

        if self.provider != "openai":
            return None

        if not self.api_key or not self.model:
            return None

        try:
            from openai import OpenAI
        except ImportError:
            return None

        try:
            import base64

            encoded = base64.b64encode(
                image_path.read_bytes()
            ).decode("utf-8")

            client = OpenAI(
                api_key=self.api_key,
            )

            response = client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Extract the transaction amount "
                                    "from this financial image. "
                                    "Return JSON only with keys: "
                                    "amount, currency, confidence. "
                                    "Do not guess. If no amount is "
                                    "visible, return null."
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": (
                                    "data:image/png;base64,"
                                    f"{encoded}"
                                ),
                            },
                        ],
                    }
                ],
            )

            text = getattr(
                response,
                "output_text",
                "",
            )

            data = json.loads(text)

            amount = data.get("amount")
            confidence = float(
                data.get("confidence", 0)
            )

            if amount is None:
                return None

            if confidence < 0.60:
                return None

            return {
                "amount": float(amount),
                "currency": data.get("currency"),
                "confidence": confidence,
            }

        except Exception:
            return None