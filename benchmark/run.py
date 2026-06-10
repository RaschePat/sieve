"""벤치마크 — 동일 태스크 셋을 두 모드로 돌려 비용/지연/라우팅을 비교한다.

  모드 A (cloud-only): 모든 요청을 강제로 클라우드(Claude)로 보낸다 (x-route: cloud)
  모드 B (hybrid):     라우터의 분류기 결정에 맡긴다 (단순 작업은 로컬)

두 모드의 총비용을 비교해 "하이브리드가 클라우드 대비 N% 절감"을 산출한다.
README에 붙일 수 있는 마크다운 표를 출력한다.

전제: 라우터 서버가 떠 있어야 한다 (uvicorn app.main:app --port 8400)
      cloud-only 모드는 ANTHROPIC_API_KEY 가 필요하다.

사용법:
  python benchmark/run.py                 # 두 모드 모두
  python benchmark/run.py --hybrid-only   # 클라우드 키 없이 하이브리드만
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROUTER_URL = "http://localhost:8400/v1/chat/completions"
TASKS_PATH = Path(__file__).parent / "tasks.jsonl"


def load_tasks() -> list[dict]:
    with open(TASKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_one(task: dict, force_route: str | None) -> dict | None:
    headers = {"Content-Type": "application/json"}
    if force_route:
        headers["x-route"] = force_route
    payload = {"messages": [{"role": "user", "content": task["content"]}], "max_tokens": 600}
    try:
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(ROUTER_URL, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["router"]
    except httpx.HTTPError as e:
        print(f"  ! {task['id']} 실패: {e}", file=sys.stderr)
        return None


def run_mode(name: str, tasks: list[dict], force_route: str | None) -> dict:
    print(f"\n▶ {name} 모드 실행 중 ({len(tasks)}건)...")
    agg = {"cost": 0.0, "saved": 0.0, "local": 0, "cloud": 0, "latency": 0, "fail": 0}
    started = time.monotonic()
    for t in tasks:
        r = run_one(t, force_route)
        if r is None:
            agg["fail"] += 1
            continue
        agg["cost"] += r["cost_usd"]
        agg["saved"] += r["saved_usd"]
        agg["latency"] += r["latency_ms"]
        agg[r["route"]] += 1
        mark = "L" if r["route"] == "local" else "C"
        print(f"  [{mark}] {t['id']:<10} {t['kind']:<6} {r['latency_ms']:>6}ms  ${r['cost_usd']:.5f}  ({r['reason']})")
    agg["wall_s"] = round(time.monotonic() - started, 1)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid-only", action="store_true", help="클라우드 키 없이 하이브리드만 측정")
    args = ap.parse_args()

    tasks = load_tasks()
    n = len(tasks)

    hybrid = run_mode("하이브리드", tasks, force_route=None)

    if args.hybrid_only:
        print_hybrid_only(n, hybrid)
        return

    cloud = run_mode("클라우드-only", tasks, force_route="cloud")
    print_comparison(n, cloud, hybrid)


def print_hybrid_only(n: int, hybrid: dict):
    print("\n" + "=" * 60)
    print("## 하이브리드 모드 결과")
    print(f"- 총 {n}건 중 로컬 {hybrid['local']}건 / 클라우드 {hybrid['cloud']}건 (실패 {hybrid['fail']})")
    print(f"- 로컬 처리율: {hybrid['local'] / n * 100:.1f}%")
    print(f"- 실제 지출: ${hybrid['cost']:.5f}")
    print(f"- 절약 추정액: ${hybrid['saved']:.5f}")
    print(f"- 총 소요: {hybrid['wall_s']}s")
    print("\n(클라우드-only 비교는 ANTHROPIC_API_KEY 설정 후 `python benchmark/run.py`)")


def print_comparison(n: int, cloud: dict, hybrid: dict):
    saving = cloud["cost"] - hybrid["cost"]
    pct = (saving / cloud["cost"] * 100) if cloud["cost"] else 0.0

    print("\n" + "=" * 60)
    print("## 벤치마크 결과\n")
    print(f"| 지표 | 클라우드-only | 하이브리드 |")
    print(f"|---|---|---|")
    print(f"| 총 비용 (USD) | ${cloud['cost']:.5f} | ${hybrid['cost']:.5f} |")
    print(f"| 로컬 처리 건수 | 0 / {n} | {hybrid['local']} / {n} |")
    print(f"| 클라우드 호출 건수 | {cloud['cloud']} | {hybrid['cloud']} |")
    print(f"| 총 지연 합 (ms) | {cloud['latency']:,} | {hybrid['latency']:,} |")
    print(f"| 벽시계 시간 (s) | {cloud['wall_s']} | {hybrid['wall_s']} |")
    print(f"\n**절감액: ${saving:.5f} ({pct:.1f}% 절감)**")
    print(f"\n> 하이브리드는 {n}건 중 {hybrid['local']}건을 로컬에서 처리해 "
          f"클라우드 토큰 비용을 {pct:.0f}% 줄였다.")


if __name__ == "__main__":
    main()
