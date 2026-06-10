"""LLM 스마트 라우터 — OpenAI 호환 프록시 서버.

기존 코드의 base URL만 이 서버로 바꾸면 단순 작업은 로컬 LLM(Ollama),
복잡한 작업은 클라우드(Claude)로 자동 분배된다.

실행: uvicorn app.main:app --port 8400
"""

import random
import time
import uuid

import anthropic
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import config, db, judge, router
from .providers import cloud, local
from .providers.local import LocalModelError

app = FastAPI(title="LLM Smart Router")


class ChatRequest(BaseModel):
    model: str = "auto"  # 'auto' 외 값은 무시되고 라우터가 결정
    messages: list[dict]
    max_tokens: int | None = None
    stream: bool = False


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatRequest,
    background: BackgroundTasks,
    x_route: str | None = Header(default=None),
):
    if req.stream:
        raise HTTPException(400, "streaming은 아직 미지원입니다 (roadmap)")

    decision = router.decide(req.messages, force=x_route)
    started = time.monotonic()
    fallback = False

    try:
        if decision.route == "local":
            try:
                result = await local.chat(req.messages, req.max_tokens)
            except LocalModelError as e:
                # 로컬 실패 → 클라우드 폴백 (캐스케이드)
                fallback = True
                decision.reason += f" → 로컬 실패({e}), 클라우드 폴백"
                result = await cloud.chat(req.messages, req.max_tokens)
        else:
            result = await cloud.chat(req.messages, req.max_tokens)
    except cloud.CloudAuthError as e:
        raise HTTPException(502, f"클라우드 호출 실패: {e}")
    except anthropic.AuthenticationError:
        raise HTTPException(502, "클라우드 호출 실패: ANTHROPIC_API_KEY가 없거나 잘못됨")
    except anthropic.APIError as e:
        raise HTTPException(502, f"클라우드 호출 실패: {e}")

    latency_ms = int((time.monotonic() - started) * 1000)
    route_final = "cloud" if (decision.route == "cloud" or fallback) else "local"

    if route_final == "local":
        cost_usd = 0.0
        # 절약액 = 같은 토큰량을 클라우드로 처리했을 때의 추정 비용
        # (토크나이저가 달라 근사치 — README의 측정 방법론 참고)
        saved_usd = config.cloud_cost_usd(
            config.CLOUD_MODEL, result["input_tokens"], result["output_tokens"]
        )
    else:
        cost_usd = result.get("cost_usd", 0.0)
        saved_usd = 0.0

    request_id = db.log_request(
        route=route_final,
        reason=decision.reason,
        model=result["model"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        cost_usd=cost_usd,
        saved_usd=saved_usd,
        latency_ms=latency_ms,
        fallback=fallback,
    )

    # 로컬 응답을 확률적으로 샘플링해 품질 채점 (백그라운드)
    if route_final == "local" and random.random() < config.JUDGE_SAMPLE_RATE:
        user_prompt = next(
            (m["content"] for m in reversed(req.messages)
             if m.get("role") == "user" and isinstance(m.get("content"), str)),
            "",
        )
        background.add_task(judge.judge, request_id, user_prompt, result["text"])

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result["model"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["text"]},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result["input_tokens"],
            "completion_tokens": result["output_tokens"],
            "total_tokens": result["input_tokens"] + result["output_tokens"],
        },
        # OpenAI 스펙 외 확장 필드: 라우팅 투명성
        "router": {
            "route": route_final,
            "reason": decision.reason,
            "fallback": fallback,
            "cost_usd": round(cost_usd, 6),
            "saved_usd": round(saved_usd, 6),
            "latency_ms": latency_ms,
        },
    }


@app.get("/stats")
async def get_stats():
    return db.stats()


@app.get("/healthz")
async def healthz():
    return {"ok": True}


DASHBOARD_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>LLM Smart Router</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 880px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 14px; margin-top: 24px; }
  .card { border: 1px solid #e2e2e2; border-radius: 10px; padding: 16px; }
  .card .label { font-size: .78rem; color: #777; }
  .card .value { font-size: 1.5rem; font-weight: 700; margin-top: 4px; }
  .highlight .value { color: #0a7d36; }
  footer { margin-top: 32px; font-size: .8rem; color: #999; }
</style></head><body>
<h1>LLM Smart Router — 비용 대시보드</h1>
<div class="grid" id="grid"></div>
<footer>10초마다 자동 갱신 · 데이터: /stats</footer>
<script>
async function refresh() {
  const s = await (await fetch('/stats')).json();
  const cards = [
    ['총 요청', s.total_requests],
    ['로컬 처리율', (s.local_ratio * 100).toFixed(1) + '%', true],
    ['절약액 (USD)', '$' + s.total_saved_usd.toFixed(4), true],
    ['실제 지출 (USD)', '$' + s.total_cost_usd.toFixed(4)],
    ['로컬 처리', s.local_count + '건'],
    ['클라우드 처리', s.cloud_count + '건'],
    ['폴백', s.fallback_count + '건'],
    ['로컬 평균 지연', s.local_avg_latency_ms ? s.local_avg_latency_ms + 'ms' : '-'],
    ['클라우드 평균 지연', s.cloud_avg_latency_ms ? s.cloud_avg_latency_ms + 'ms' : '-'],
    ['품질 점수 (judge)', s.avg_judge_score ? s.avg_judge_score + ' / 5 (' + s.judged_count + '건)' : '-', true],
  ];
  document.getElementById('grid').innerHTML = cards.map(([label, value, hl]) =>
    `<div class="card ${hl ? 'highlight' : ''}"><div class="label">${label}</div><div class="value">${value}</div></div>`
  ).join('');
}
refresh(); setInterval(refresh, 10000);
</script></body></html>"""


@app.get("/dashboard")
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)
