"""한국어 가드 A/B 테스트 — 가드 off vs on 의 '중국어 혼용'을 정량 비교한다.

Qwen 같은 다국어 모델은 한국어 요청에 중국어 한자(漢字)를 섞는 경향이 있다.
한글(U+AC00–D7A3)과 중국어 한자(U+4E00–9FFF)는 유니코드 블록이 다르므로,
응답에 섞인 '한자 개수'를 세면 혼용 정도를 객관적·결정론적으로 측정할 수 있다.

로컬로 라우팅되는 태스크만 골라 가드 off/on 두 번 처리하고 한자 개수를 비교한다.
서버 불필요(app 모듈 직접 호출), 크레딧 불필요(로컬 생성만 사용).
judge 점수는 크레딧이 있으면 보조 지표로 함께 출력한다.

  python benchmark/korean_guard_ab.py
"""

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import judge, router  # noqa: E402
from app.providers import local  # noqa: E402

TASKS_PATH = Path(__file__).parent / "tasks.jsonl"

# 중국어 한자 블록 (한글·일본어 가나 제외). 한국어 응답에 이게 있으면 혼용으로 본다.
_HANJA_RE = re.compile(r"[一-鿿]")


def count_hanja(text: str) -> int:
    return len(_HANJA_RE.findall(text))


def local_tasks() -> list[dict]:
    tasks = [json.loads(l) for l in open(TASKS_PATH, encoding="utf-8") if l.strip()]
    out = []
    for t in tasks:
        d = router.decide([{"role": "user", "content": t["content"]}])
        if d.route == "local":
            out.append(t)
    return out


async def run_variant(task: dict, guard: bool) -> dict:
    msgs = [{"role": "user", "content": task["content"]}]
    result = await local.chat(msgs, max_tokens=600, korean_guard=guard)
    text = result["text"]
    score = None
    try:
        score, _ = await judge.score_one(task["content"], text)
    except Exception:  # noqa: BLE001 — 크레딧 없으면 judge 생략, 한자 지표만 사용
        pass
    return {"text": text, "hanja": count_hanja(text), "score": score}


async def main():
    tasks = local_tasks()
    print(f"로컬 태스크 {len(tasks)}건: 가드 OFF/ON 비교 (중국어 한자 개수 측정)\n")

    rows = []
    for t in tasks:
        off = await run_variant(t, guard=False)
        on = await run_variant(t, guard=True)
        rows.append((t["id"], t["kind"], off, on))
        flag = "⚠️ 혼용→해결" if off["hanja"] > 0 and on["hanja"] == 0 else ""
        print(f"  {t['id']:<10} {t['kind']:<6} 한자 {off['hanja']:>3} → {on['hanja']:>3}  {flag}")

    n = len(rows)
    off_total = sum(r[2]["hanja"] for r in rows)
    on_total = sum(r[3]["hanja"] for r in rows)
    off_mixed = sum(1 for r in rows if r[2]["hanja"] > 0)
    on_mixed = sum(1 for r in rows if r[3]["hanja"] > 0)
    has_judge = all(r[2]["score"] is not None for r in rows)

    print("\n" + "=" * 60)
    print("## 한국어 가드 A/B 결과\n")
    header = "| 태스크 | 유형 | 한자 OFF | 한자 ON |"
    sep = "|---|---|---|---|"
    if has_judge:
        header += " judge OFF | judge ON |"
        sep += "---|---|"
    print(header)
    print(sep)
    for tid, kind, off, on in rows:
        line = f"| {tid} | {kind} | {off['hanja']} | {on['hanja']} |"
        if has_judge:
            line += f" {off['score']}/5 | {on['score']}/5 |"
        print(line)
    print("")
    print(f"- **중국어 혼용 응답 수:** {off_mixed}/{n} → **{on_mixed}/{n}**")
    print(f"- **총 한자 개수:** {off_total} → **{on_total}** "
          f"({'-100%' if off_total and on_total == 0 else f'{(on_total - off_total) / off_total * 100:+.0f}%' if off_total else 'n/a'})")
    if has_judge:
        ab = sum(r[2]["score"] for r in rows) / n
        aa = sum(r[3]["score"] for r in rows) / n
        print(f"- **judge 평균 품질:** {ab:.2f}/5 → **{aa:.2f}/5** ({aa - ab:+.2f})")
    else:
        print("- _judge 채점은 크레딧 부족으로 생략 — 한자 지표만 측정됨_")


if __name__ == "__main__":
    asyncio.run(main())
