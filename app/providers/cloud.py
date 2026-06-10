"""클라우드 LLM 프로바이더 — Anthropic Claude API.

클라우드로 라우팅되는 요청은 고난도 작업이므로 adaptive thinking을 켠다.
"""

import os

import anthropic

from .. import config

_client: anthropic.AsyncAnthropic | None = None


class CloudAuthError(Exception):
    """클라우드 인증 정보 미설정 — 라우터가 502로 변환해 명확히 안내한다."""


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        # 키가 아예 없으면 SDK가 모호한 TypeError를 던지므로 사전에 잡아 명확히 알린다
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            raise CloudAuthError("ANTHROPIC_API_KEY 미설정 — 클라우드 라우팅 불가")
        _client = anthropic.AsyncAnthropic()
    return _client


def _split_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """OpenAI 스타일 메시지에서 system을 분리하고 Anthropic 형식으로 변환."""
    system_parts: list[str] = []
    converted: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            converted.append({"role": role, "content": content})
    return ("\n\n".join(system_parts) or None), converted


async def chat(messages: list[dict], max_tokens: int | None = None) -> dict:
    """Claude에 요청을 보내고 {text, input_tokens, output_tokens, cost_usd}를 반환."""
    system, converted = _split_messages(messages)

    kwargs: dict = {
        "model": config.CLOUD_MODEL,
        "max_tokens": max_tokens or 16000,
        "thinking": {"type": "adaptive"},
        "messages": converted,
    }
    if system:
        kwargs["system"] = system

    response = await get_client().messages.create(**kwargs)

    text = next((b.text for b in response.content if b.type == "text"), "")
    usage = response.usage
    input_tokens = (
        usage.input_tokens
        + (usage.cache_creation_input_tokens or 0)
        + (usage.cache_read_input_tokens or 0)
    )
    return {
        "text": text,
        "model": config.CLOUD_MODEL,
        "input_tokens": input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": config.cloud_cost_usd(config.CLOUD_MODEL, usage.input_tokens, usage.output_tokens),
    }
