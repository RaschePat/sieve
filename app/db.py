"""SQLite 기반 요청 로깅. 모든 라우팅 결정과 비용을 기록한다 — 이 데이터가 절약 증명의 근거."""

import sqlite3
import threading
import time

from . import config

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    route TEXT NOT NULL,            -- 'local' | 'cloud'
    reason TEXT NOT NULL,           -- 라우팅 결정 사유
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,         -- 실제 발생 비용 (local이면 0)
    saved_usd REAL NOT NULL,        -- local 처리로 절약한 추정 클라우드 비용
    latency_ms INTEGER NOT NULL,
    fallback INTEGER NOT NULL DEFAULT 0,  -- local 실패 후 cloud 폴백 여부
    judge_score INTEGER,            -- 1~5, 샘플링된 요청만
    judge_rationale TEXT
);
"""


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _conn.execute(SCHEMA)
        _conn.commit()
    return _conn


def log_request(
    *,
    route: str,
    reason: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    saved_usd: float,
    latency_ms: int,
    fallback: bool = False,
) -> int:
    with _lock:
        conn = get_conn()
        cur = conn.execute(
            "INSERT INTO requests (ts, route, reason, model, input_tokens, output_tokens,"
            " cost_usd, saved_usd, latency_ms, fallback)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), route, reason, model, input_tokens, output_tokens,
             cost_usd, saved_usd, latency_ms, int(fallback)),
        )
        conn.commit()
        return cur.lastrowid


def set_judge_result(request_id: int, score: int, rationale: str) -> None:
    with _lock:
        conn = get_conn()
        conn.execute(
            "UPDATE requests SET judge_score = ?, judge_rationale = ? WHERE id = ?",
            (score, rationale, request_id),
        )
        conn.commit()


def stats() -> dict:
    with _lock:
        conn = get_conn()
        row = conn.execute(
            """
            SELECT
                COUNT(*)                                              AS total,
                SUM(CASE WHEN route = 'local' THEN 1 ELSE 0 END)      AS local_count,
                SUM(CASE WHEN route = 'cloud' THEN 1 ELSE 0 END)      AS cloud_count,
                SUM(cost_usd)                                         AS total_cost,
                SUM(saved_usd)                                        AS total_saved,
                SUM(CASE WHEN fallback = 1 THEN 1 ELSE 0 END)         AS fallback_count,
                AVG(CASE WHEN route = 'local' THEN latency_ms END)    AS local_avg_latency,
                AVG(CASE WHEN route = 'cloud' THEN latency_ms END)    AS cloud_avg_latency,
                AVG(judge_score)                                      AS avg_judge_score,
                COUNT(judge_score)                                    AS judged_count
            FROM requests
            """
        ).fetchone()

    total = row[0] or 0
    local_count = row[1] or 0
    return {
        "total_requests": total,
        "local_count": local_count,
        "cloud_count": row[2] or 0,
        "local_ratio": round(local_count / total, 3) if total else 0.0,
        "total_cost_usd": round(row[3] or 0.0, 4),
        "total_saved_usd": round(row[4] or 0.0, 4),
        "fallback_count": row[5] or 0,
        "local_avg_latency_ms": round(row[6]) if row[6] else None,
        "cloud_avg_latency_ms": round(row[7]) if row[7] else None,
        "avg_judge_score": round(row[8], 2) if row[8] else None,
        "judged_count": row[9] or 0,
    }
