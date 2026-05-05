"""Optional AI sentiment adapter used to adjust signal confidence."""

from __future__ import annotations

import re
import time
from typing import Optional

import requests

from config import CONFIG
from core.resilience import TokenBucketLimiter, retry_delay_seconds
from core.signal import Direction, Signal
from utils.logger import get_logger

log = get_logger("AISentiment")


class AISentimentEngine:
    """Provider-agnostic sentiment scorer with bounded confidence adjustment."""

    def __init__(self):
        self._enabled = bool(CONFIG.ai.sentiment_enabled)
        self._cache_ttl = max(30, int(CONFIG.ai.sentiment_cache_seconds))
        self._cache: dict[str, tuple[float, float]] = {}
        self._max_adjust = max(0.0, float(CONFIG.ai.sentiment_max_adjustment))
        self._timeout = max(2, int(CONFIG.ai.sentiment_timeout_seconds))
        self._provider = str(CONFIG.ai.provider).lower()
        self._retries = max(0, int(CONFIG.api.retry_attempts))
        self._base_delay = float(CONFIG.api.backoff_base_seconds)
        self._cap_delay = float(CONFIG.api.backoff_cap_seconds)
        self._limiter = TokenBucketLimiter(CONFIG.api.rate_limit_per_minute)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def confidence_adjustment(
        self,
        symbol: str,
        signal: Signal,
        regime: str,
        context: dict,
    ) -> float:
        if not self._enabled or self._max_adjust <= 0:
            return 0.0

        cache_key = f"{symbol}:{signal.direction.value}:{regime}"
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached and now - cached[0] < self._cache_ttl:
            return cached[1]

        score = self._score_sentiment(symbol, regime, context)
        if score is None:
            return 0.0

        directional_score = score if signal.direction == Direction.LONG else -score
        adjustment = max(-self._max_adjust, min(self._max_adjust, directional_score * self._max_adjust))
        self._cache[cache_key] = (now, adjustment)
        return adjustment

    def _score_sentiment(self, symbol: str, regime: str, context: dict) -> Optional[float]:
        prompt = (
            "Return exactly one number between -1 and 1. "
            "Positive means bullish sentiment, negative means bearish sentiment. "
            f"Symbol={symbol}. Regime={regime}. Context={context}."
        )

        for attempt in range(self._retries + 1):
            self._limiter.acquire()
            score = self._query_provider(prompt)
            if score is not None:
                return score
            if attempt < self._retries:
                time.sleep(retry_delay_seconds(attempt, self._base_delay, self._cap_delay))

        return None

    def _query_provider(self, prompt: str) -> Optional[float]:
        try:
            if self._provider == "openai":
                return self._query_openai(prompt)
            if self._provider == "gemini":
                return self._query_gemini(prompt)
            return self._query_glm(prompt)
        except Exception as e:
            log.debug("AI sentiment query failed: %s", e)
            return None

    def _query_openai(self, prompt: str) -> Optional[float]:
        key = CONFIG.ai.openai_api_key
        if not key:
            return None

        url = f"{CONFIG.ai.openai_base_url.rstrip('/')}/chat/completions"
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": CONFIG.ai.openai_model,
                "temperature": CONFIG.ai.openai_temperature,
                "max_tokens": min(24, int(CONFIG.ai.openai_max_tokens)),
                "messages": [
                    {"role": "system", "content": "You output only a numeric score."},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=self._timeout,
        )
        if resp.status_code != 200:
            return None
        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        return self._extract_score(content)

    def _query_glm(self, prompt: str) -> Optional[float]:
        key = CONFIG.ai.glm_api_key
        if not key:
            return None

        url = f"{CONFIG.ai.glm_base_url.rstrip('/')}/chat/completions"
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": CONFIG.ai.glm_model,
                "temperature": CONFIG.ai.glm_temperature,
                "max_tokens": min(24, int(CONFIG.ai.glm_max_tokens)),
                "messages": [
                    {"role": "system", "content": "You output only a numeric score."},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=self._timeout,
        )
        if resp.status_code != 200:
            return None
        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        return self._extract_score(content)

    def _query_gemini(self, prompt: str) -> Optional[float]:
        key = CONFIG.ai.gemini_api_key
        if not key:
            return None

        model = str(CONFIG.ai.gemini_model).removeprefix("models/")
        base_url = CONFIG.ai.gemini_base_url.rstrip("/")
        url = f"{base_url}/models/{model}:generateContent"
        resp = requests.post(
            url,
            params={"key": key},
            headers={"Content-Type": "application/json"},
            json={
                "generationConfig": {
                    "temperature": CONFIG.ai.gemini_temperature,
                    "maxOutputTokens": min(24, int(CONFIG.ai.gemini_max_tokens)),
                },
                "contents": [{"parts": [{"text": prompt}]}],
            },
            timeout=self._timeout,
        )
        if resp.status_code != 200:
            return None

        candidates = resp.json().get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        text = parts[0].get("text", "") if parts else ""
        return self._extract_score(text)

    @staticmethod
    def _extract_score(text: str) -> Optional[float]:
        match = re.search(r"-?\d+(?:\.\d+)?", text or "")
        if not match:
            return None
        try:
            value = float(match.group(0))
        except Exception:
            return None
        return max(-1.0, min(1.0, value))
