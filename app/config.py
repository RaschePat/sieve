"""환경 변수 기반 설정과 모델 단가표."""

import os

from dotenv import load_dotenv

# 프로젝트 루트의 .env 를 자동 로드 (이미 export된 환경변수가 우선)
load_dotenv()

CLOUD_MODEL = os.getenv("CLOUD_MODEL", "claude-opus-4-8")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-opus-4-8")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "Qwen2.5-7B-Instruct-4bit")

# 로컬 LLM은 OpenAI 호환 엔드포인트로 호출 (oMLX, LM Studio, vLLM, Ollama /v1 등 호환)
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "test123")

# 로컬 모델의 한국어 입력 시 의도치 않은 언어 혼용(중국어 등)을 막는 가드 (기본 켜짐)
LOCAL_KOREAN_GUARD = os.getenv("LOCAL_KOREAN_GUARD", "1") == "1"
DB_PATH = os.getenv("ROUTER_DB_PATH", "router.db")
JUDGE_SAMPLE_RATE = float(os.getenv("JUDGE_SAMPLE_RATE", "0.1"))

# USD per 1M tokens (2026-06 기준, platform.claude.com/docs 단가)
PRICING = {
    "claude-fable-5": {"input": 10.00, "output": 50.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}


def cloud_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """토큰 수 → USD 비용. 단가표에 없는 모델은 CLOUD_MODEL 단가로 추정."""
    price = PRICING.get(model) or PRICING[CLOUD_MODEL]
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
