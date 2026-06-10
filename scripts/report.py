"""DB 로그 → 마크다운 리포트 생성기.

라우터가 실제 처리한 모든 요청을 집계해 비용/품질 리포트를 만든다.
  python scripts/report.py            # 표준출력
  python scripts/report.py > REPORT.md
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, db  # noqa: E402


def reason_breakdown(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(
        "SELECT route, reason, COUNT(*) c, SUM(saved_usd) s "
        "FROM requests GROUP BY route, reason ORDER BY c DESC"
    ).fetchall()


def judge_rows(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(
        "SELECT judge_score, judge_rationale, model FROM requests "
        "WHERE judge_score IS NOT NULL ORDER BY id DESC LIMIT 20"
    ).fetchall()


def main():
    s = db.stats()
    conn = db.get_conn()

    out: list[str] = []
    w = out.append

    w("# LLM Smart Router — 운영 리포트\n")
    if s["total_requests"] == 0:
        w("_아직 처리된 요청이 없습니다. 라우터로 요청을 보낸 뒤 다시 실행하세요._")
        print("\n".join(out))
        return

    w("## 요약\n")
    w(f"- **총 요청:** {s['total_requests']}건")
    w(f"- **로컬 처리율:** {s['local_ratio'] * 100:.1f}% "
      f"(로컬 {s['local_count']} / 클라우드 {s['cloud_count']})")
    w(f"- **실제 지출:** ${s['total_cost_usd']:.4f}")
    w(f"- **절약 추정액:** ${s['total_saved_usd']:.4f}")
    if s["total_cost_usd"] + s["total_saved_usd"] > 0:
        denom = s["total_cost_usd"] + s["total_saved_usd"]
        w(f"- **절감율:** {s['total_saved_usd'] / denom * 100:.1f}% "
          f"(전부 클라우드로 처리했을 경우 대비)")
    w(f"- **폴백:** {s['fallback_count']}건 (로컬 실패 → 클라우드)")
    w("")

    w("## 지연 시간\n")
    w("| 경로 | 평균 지연 |")
    w("|---|---|")
    w(f"| 로컬 | {s['local_avg_latency_ms'] or '-'} ms |")
    w(f"| 클라우드 | {s['cloud_avg_latency_ms'] or '-'} ms |")
    w("")

    w("## 라우팅 사유별 분포\n")
    w("| 경로 | 사유 | 건수 | 절약액(USD) |")
    w("|---|---|---|---|")
    for route, reason, c, saved in reason_breakdown(conn):
        w(f"| {route} | {reason} | {c} | ${(saved or 0):.4f} |")
    w("")

    w("## 품질 검증 (LLM-as-judge)\n")
    if s["avg_judge_score"] is None:
        w(f"_샘플링된 채점 없음 (JUDGE_SAMPLE_RATE={config.JUDGE_SAMPLE_RATE}). "
          "로컬 응답 일부가 채점되면 여기에 표시됩니다._")
    else:
        w(f"- **평균 품질 점수:** {s['avg_judge_score']} / 5 "
          f"({s['judged_count']}건 채점, 로컬 응답 샘플)")
        w("")
        w("| 점수 | 평가 | 모델 |")
        w("|---|---|---|")
        for score, rationale, model in judge_rows(conn):
            w(f"| {score}/5 | {rationale} | {model} |")
    w("")

    w("---")
    w(f"_클라우드 단가 기준: {config.CLOUD_MODEL} "
      f"(${config.PRICING.get(config.CLOUD_MODEL, {}).get('input', '?')}/"
      f"${config.PRICING.get(config.CLOUD_MODEL, {}).get('output', '?')} per 1M tokens)_")

    print("\n".join(out))


if __name__ == "__main__":
    main()
