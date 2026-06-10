"""LLM-as-judge 품질 검증.

로컬 모델이 처리한 응답을 샘플링해서 클라우드 모델로 채점한다.
"비용은 줄였지만 품질은 유지됐다"를 증명하는 핵심 장치.
"""

import json

import anthropic

from . import config, db

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": "응답 품질 점수. 5=완벽, 4=좋음, 3=쓸만함, 2=미흡, 1=실패",
        },
        "rationale": {"type": "string", "description": "한 문장 평가 근거"},
    },
    "required": ["score", "rationale"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = (
    "당신은 LLM 응답 품질 평가자입니다. 사용자 요청과 모델 응답을 보고 "
    "응답이 요청을 얼마나 정확하고 충실하게 수행했는지 1~5점으로 평가하세요. "
    "장황함보다 정확성과 요청 충족도를 우선합니다."
)

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


async def score_one(user_prompt: str, answer: str) -> tuple[int, str]:
    """응답을 채점해 (점수, 근거)를 반환한다. DB에 쓰지 않음 — 순수 채점."""
    response = await _get_client().messages.create(
        model=config.JUDGE_MODEL,
        max_tokens=1024,
        system=JUDGE_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"## 사용자 요청\n{user_prompt[:4000]}\n\n"
                f"## 모델 응답\n{answer[:4000]}\n\n"
                "이 응답을 평가하세요."
            ),
        }],
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    result = json.loads(text)
    return result["score"], result["rationale"]


async def judge(request_id: int, user_prompt: str, answer: str) -> None:
    """백그라운드에서 실행. 채점 후 DB에 기록. 실패해도 본 요청에는 영향 없음."""
    try:
        score, rationale = await score_one(user_prompt, answer)
        db.set_judge_result(request_id, score, rationale)
    except Exception as e:  # noqa: BLE001 — 채점 실패는 로그만 남기고 무시
        print(f"[judge] request {request_id} 채점 실패: {e}")
