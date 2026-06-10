"""라우팅 결정 로직.

휴리스틱 기반 분류기: 요청 텍스트에서 작업 유형을 추정해 local/cloud를 결정한다.
모든 결정은 사람이 읽을 수 있는 reason 문자열과 함께 반환되어 DB에 기록된다.
"""

import re
from dataclasses import dataclass

# 로컬 7B 모델이 충분히 잘하는 정형 작업 (한국어/영어)
LOCAL_TASK_PATTERNS = [
    (r"요약|정리해|간추려|summar(y|ize|ise)|tl;?dr", "요약"),
    (r"번역|translat(e|ion)", "번역"),
    (r"분류|classif(y|ication)|카테고리|label|태그|tag(ging)?", "분류/태깅"),
    (r"추출|extract|뽑아", "정보 추출"),
    (r"포맷|format|json으로|markdown으로|표로|convert", "포맷 변환"),
    (r"제목|title|이름 지어|네이밍|naming", "제목/네이밍"),
    (r"맞춤법|오타|교정|proofread|문법|grammar", "교정"),
    (r"감정|sentiment|긍정|부정", "감정 분석"),
    (r"키워드|keyword", "키워드 추출"),
]

# 추론/생성 난이도가 높아 클라우드로 보내야 하는 작업
CLOUD_TASK_PATTERNS = [
    (r"코드\s*리뷰|code\s*review|리팩토링|refactor", "코드 리뷰/리팩토링"),
    # 디버깅: 에러 이름, "왜 ~나는지", 원인/수정 의도
    (r"디버그|debug|버그|왜\s*안\s*되|stack\s*trace|traceback"
     r"|왜.{0,12}(나는지|발생|생기|뜨는|안)|(원인|수정\s*방법).{0,6}(설명|분석|찾|알려)"
     r"|[A-Za-z]*(Error|Exception)\b|예외가?\s*(발생|나)", "디버깅"),
    (r"설계|아키텍처|architect|architecture|구조를?\s*잡", "설계"),
    # 코드 생성: (함수/코드/클래스 …) ↔ (작성/구현/만들) 양방향 매칭
    (r"(함수|코드|클래스|메서드|스크립트|프로그램|api)\s*(를|을)?\s*(작성|구현|만들)"
     r"|(작성|구현|만들어?)\s*\S{0,6}(함수|코드|클래스|메서드|스크립트|프로그램)"
     r"|implement|write\s+(a\s+)?(function|class|code|script)", "코드 생성"),
    (r"증명|prove|수학|논리적으로|step[- ]by[- ]step", "복잡한 추론"),
    (r"비교\s*분석|평가해|trade-?off|장단점.*분석", "비교 분석"),
    (r"전략|기획|계획\s*세워|roadmap", "전략/기획"),
]

# 이 길이를 넘는 입력은 로컬 7B의 처리 품질/속도가 급격히 떨어짐
MAX_LOCAL_INPUT_CHARS = 12_000


@dataclass
class RouteDecision:
    route: str   # 'local' | 'cloud'
    reason: str


def _flatten_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(b.get("text", "") for b in content if isinstance(b, dict))
    return "\n".join(parts)


def decide(messages: list[dict], force: str | None = None) -> RouteDecision:
    """메시지를 보고 라우팅을 결정한다. force('local'|'cloud')로 강제 지정 가능."""
    if force in ("local", "cloud"):
        return RouteDecision(force, f"클라이언트 강제 지정 (x-route: {force})")

    text = _flatten_text(messages)
    lower = text.lower()

    # 1) 클라우드 트리거가 하나라도 있으면 클라우드 (안전 우선)
    for pattern, label in CLOUD_TASK_PATTERNS:
        if re.search(pattern, lower):
            return RouteDecision("cloud", f"고난도 작업 감지: {label}")

    # 2) 코드 블록이 포함된 긴 입력은 코드 이해가 필요할 가능성이 높음
    if "```" in text and len(text) > 2_000:
        return RouteDecision("cloud", "긴 코드 블록 포함 — 코드 이해 필요 가능성")

    # 3) 입력이 너무 길면 로컬 모델 품질 저하
    if len(text) > MAX_LOCAL_INPUT_CHARS:
        return RouteDecision("cloud", f"입력 {len(text):,}자 > 로컬 한도 {MAX_LOCAL_INPUT_CHARS:,}자")

    # 4) 정형 작업 패턴이 있으면 로컬
    for pattern, label in LOCAL_TASK_PATTERNS:
        if re.search(pattern, lower):
            return RouteDecision("local", f"정형 작업 감지: {label}")

    # 5) 짧은 일반 질문은 로컬로 시도 (실패 시 폴백)
    if len(text) < 500:
        return RouteDecision("local", "짧은 일반 요청 — 로컬 우선 시도")

    # 6) 그 외 애매한 경우는 클라우드 (품질 우선)
    return RouteDecision("cloud", "분류 불확실 — 품질 우선으로 클라우드")
