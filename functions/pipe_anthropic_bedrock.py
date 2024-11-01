"""Pipeline for Bedrock Anthropic."""  # noqa: INP001

import os
from collections.abc import Generator
from typing import Any

import anthropic
import pydantic
from open_webui.utils import misc as open_webui_misc


class Pipe:
    """Anthropic Bedrock Connector."""

    class Valves(pydantic.BaseModel):
        """Variables exposed via the admin UI."""

        AWS_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY")
        AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_KEY")
        AWS_REGION: str = os.getenv("AWS_REGION")

    def __init__(self) -> None:
        """Initialize the pipe."""
        self.type = "manifold"
        self.name = "Anthropic - "
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:
        """List of models to include."""
        return [
            {
                "id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "name": "Claude 3.5 Sonnet V2",
            },
        ]

    def process_image(self, image_data: dict[str, Any]) -> dict[str, Any]:
        """Converts images into a format accepted by Claude."""
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        return {
            "type": "image",
            "source": {"type": "url", "url": image_data["image_url"]["url"]},
        }

    def pipe(self, body: dict) -> Generator[str, None, None]:
        """Main runner of the pipe.

        Args:
            body: The request as provided by Open WebUI.

        Returns:
            A string or iterable string.
        """
        system_message, messages = open_webui_misc.pop_system_message(body["messages"])
        if system_message is None:
            system_message = {"content": ""}

        processed_messages = []
        image_count = 0
        total_image_size = 0

        for message in messages:
            processed_content = []
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item["type"] == "text":
                        processed_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        if image_count >= 5:  # noqa: PLR2004
                            msg = "Maximum of 5 images per API call exceeded"
                            raise ValueError(msg)

                        processed_image = self.process_image(item)
                        processed_content.append(processed_image)

                        if processed_image["source"]["type"] == "base64":
                            image_size = len(processed_image["source"]["data"]) * 3 / 4
                        else:
                            image_size = 0

                        total_image_size += image_size
                        if total_image_size > 100 * 1024 * 1024:
                            msg = "Total size of images exceeds 100 MB limit"
                            raise ValueError(msg)
                        image_count += 1
            else:
                processed_content = [
                    {"type": "text", "text": message.get("content", "")},
                ]

            processed_messages.append(
                {"role": message["role"], "content": processed_content},
            )

        client = anthropic.AnthropicBedrock(
            aws_access_key=self.valves.AWS_ACCESS_KEY,
            aws_secret_key=self.valves.AWS_SECRET_ACCESS_KEY,
            aws_region=self.valves.AWS_REGION,
        )

        model = body["model"][body["model"].find(".") + 1 :]
        with client.messages.stream(
            system=system_message["content"],
            model=model,
            messages=processed_messages,
            max_tokens=8192,
        ) as stream:
            yield from stream.text_stream
