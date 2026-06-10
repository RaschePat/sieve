"""로컬 LLM 프로바이더 — OpenAI 호환 엔드포인트(/v1/chat/completions) 호출.

oMLX, LM Studio, vLLM, llama.cpp server, Ollama(/v1) 등 OpenAI 호환 서버라면
LOCAL_BASE_URL / LOCAL_API_KEY 만 바꿔 그대로 쓸 수 있다.
"""

import re

import httpx

from .. import config

# 한국어 입력 시 Qwen 등 다국어 모델의 중국어 혼용을 막는 시스템 프롬프트.
# 번역/작성을 명시적으로 요청한 경우는 그 언어로 답하도록 예외를 둔다.
KOREAN_GUARD_PROMPT = (
    "사용자의 입력이 한국어이면 한국어로만 답하세요. "
    "단, 사용자가 특정 언어로의 번역이나 작성을 명시적으로 요청한 경우에는 그 언어로 답하세요. "
    "어떤 경우에도 요청하지 않은 언어(특히 중국어 한자)를 답변에 섞지 마세요. "
    "자연스럽고 일관된 단일 언어로 작성하세요."
)

_KOREAN_RE = re.compile(r"[가-힣]")


class LocalModelError(Exception):
    """로컬 모델 호출 실패. 호출자는 클라우드 폴백을 수행한다."""


def _has_korean(messages: list[dict]) -> bool:
    for m in messages:
        if _KOREAN_RE.search(_to_text(m.get("content", ""))):
            return True
    return False


def _apply_korean_guard(messages: list[dict]) -> list[dict]:
    """한국어 입력이면 가드 시스템 프롬프트를 맨 앞에 주입한다."""
    if not _has_korean(messages):
        return messages
    # 기존 system 메시지가 있으면 그 뒤에, 없으면 맨 앞에 가드를 넣는다
    guard = {"role": "system", "content": KOREAN_GUARD_PROMPT}
    if messages and messages[0].get("role") == "system":
        return [messages[0], guard, *messages[1:]]
    return [guard, *messages]


async def chat(
    messages: list[dict], max_tokens: int | None = None, korean_guard: bool | None = None
) -> dict:
    """로컬 서버에 채팅 요청을 보내고 {text, input_tokens, output_tokens}를 반환.

    korean_guard=None이면 config.LOCAL_KOREAN_GUARD 기본값을 따른다.
    A/B 테스트에서 명시적으로 True/False를 넘겨 가드를 토글할 수 있다.
    """
    use_guard = config.LOCAL_KOREAN_GUARD if korean_guard is None else korean_guard
    effective = _apply_korean_guard(messages) if use_guard else messages

    payload = {
        "model": config.LOCAL_MODEL,
        "messages": [
            {"role": m["role"], "content": _to_text(m.get("content", ""))}
            for m in effective
        ],
        "stream": False,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    headers = {"Content-Type": "application/json"}
    if config.LOCAL_API_KEY:
        headers["Authorization"] = f"Bearer {config.LOCAL_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{config.LOCAL_BASE_URL}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        raise LocalModelError(f"로컬 서버 호출 실패: {e}") from e

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LocalModelError(f"예상치 못한 응답 형식: {data}") from e

    if not text or not text.strip():
        raise LocalModelError("로컬 서버가 빈 응답을 반환")

    usage = data.get("usage") or {}
    return {
        "text": text,
        "model": config.LOCAL_MODEL,
        # OpenAI 표준 필드. oMLX는 input_tokens/output_tokens도 함께 보고함
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
    }


def _to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)
