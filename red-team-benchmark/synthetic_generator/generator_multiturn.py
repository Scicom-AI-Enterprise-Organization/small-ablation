import asyncio
import json
import logging
import os
import random
from typing import Any

import aiohttp

from .prompts_multiturn import SYSTEM_PROMPT_MULTITURN, USER_PROMPT_TEMPLATE_MULTITURN
from .schema import LANGUAGES, MULTITURN_REQUIRED_FIELDS

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://serverlessgpu.aies.scicom.dev/tm-fleet/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3.5-122B-A10B"


class MultiTurnGenerator:
    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: int = 180,
    ):
        self.api_url = api_url
        self.model = model
        self.api_key = api_key or os.environ.get("SGPU_API_KEY", "")
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def _call_api(self, session: aiohttp.ClientSession, count: int, language: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_MULTITURN},
                {"role": "user", "content": USER_PROMPT_TEMPLATE_MULTITURN.format(count=count, language=language)},
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
        issues = [f for f in MULTITURN_REQUIRED_FIELDS if f not in entry]

        conversation = entry.get("conversation", [])
        if not isinstance(conversation, list) or len(conversation) < 4:
            issues.append("conversation must be a list with at least 4 turns")
        else:
            for i, turn in enumerate(conversation):
                if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
                    issues.append(f"conversation[{i}] missing role or content")
                elif turn["role"] not in ("user", "assistant"):
                    issues.append(f"conversation[{i}] has invalid role: {turn['role']!r}")

        attack_idx = entry.get("attack_turn_index")
        if isinstance(attack_idx, int) and isinstance(conversation, list):
            if attack_idx < 0 or attack_idx >= len(conversation):
                issues.append(f"attack_turn_index {attack_idx} is out of range for conversation length {len(conversation)}")
            elif conversation[attack_idx].get("role") != "user":
                issues.append(f"attack_turn_index {attack_idx} must point to a user turn")

        return issues

    async def generate(
        self,
        session: aiohttp.ClientSession,
        count: int = 10,
        retries: int = 2,
    ) -> tuple[list[dict[str, Any]], str]:
        language = random.choice(LANGUAGES)
        for attempt in range(retries + 1):
            try:
                logger.info(
                    "Calling API (attempt %d/%d) for %d multi-turn entries [lang=%s]...",
                    attempt + 1, retries + 1, count, language,
                )
                raw = await self._call_api(session, count, language)
                entries = self._parse_response(raw)

                issues = [
                    f"Entry {i}: {problem}"
                    for i, e in enumerate(entries)
                    for problem in self._validate_entry(e)
                ]
                if issues:
                    logger.warning("Validation warnings:\n" + "\n".join(issues))

                logger.info("Generated %d multi-turn entries.", len(entries))
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
