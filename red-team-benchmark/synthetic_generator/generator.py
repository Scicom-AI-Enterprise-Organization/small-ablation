import asyncio
import json
import logging
import os
import random
from typing import Any

import aiohttp

from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .schema import LANGUAGES, REQUIRED_FIELDS

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://serverlessgpu.aies.scicom.dev/tm-fleet/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3.5-122B-A10B"


class DatasetGenerator:
    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: int = 120,
    ):
        self.api_url = api_url
        self.model = model
        self.api_key = api_key or os.environ.get("SGPU_API_KEY", "")
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def _call_api(self, session: aiohttp.ClientSession, count: int, language: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(count=count, language=language)},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        async with session.post(self.api_url, json=payload, headers=headers) as response:
            response.raise_for_status()
            result = await response.json()
            return result["choices"][0]["message"]["content"]

    def _parse_response(self, raw: str) -> list[dict[str, Any]]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            from json_repair import repair_json
            logger.warning("JSON malformed — attempting repair.")
            return json.loads(repair_json(text))

    def _validate_entry(self, entry: dict) -> list[str]:
        return [f for f in REQUIRED_FIELDS if f not in entry]

    async def generate(
        self,
        session: aiohttp.ClientSession,
        count: int = 20,
        retries: int = 2,
    ) -> tuple[list[dict[str, Any]], str]:
        language = random.choice(LANGUAGES)
        for attempt in range(retries + 1):
            try:
                logger.info(
                    "Calling API (attempt %d/%d) for %d entries [lang=%s]...",
                    attempt + 1, retries + 1, count, language,
                )
                raw = await self._call_api(session, count, language)
                entries = self._parse_response(raw)

                issues = [
                    f"Entry {i}: missing fields {self._validate_entry(e)}"
                    for i, e in enumerate(entries)
                    if self._validate_entry(e)
                ]
                if issues:
                    logger.warning("Validation warnings:\n" + "\n".join(issues))

                logger.info("Generated %d entries.", len(entries))
                return entries, language

            except (json.JSONDecodeError, KeyError) as e:
                logger.error("Parse error on attempt %d: %s", attempt + 1, e)
                if attempt == retries:
                    raise
                await asyncio.sleep(2 ** attempt)

            except aiohttp.ClientResponseError as e:
                retryable = e.status in (429, 502, 503, 504)
                logger.error("HTTP %d on attempt %d%s: %s", e.status, attempt + 1, " (retrying)" if retryable and attempt < retries else "", e)
                if not retryable or attempt == retries:
                    raise
                await asyncio.sleep(2 ** attempt)

        return [], language
