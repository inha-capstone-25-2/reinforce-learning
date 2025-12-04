from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class InteractionSignal:
    """
    추천된 논문에 대해 사용자가 보인 상호작용 신호.

    - was_clicked: 클릭 여부
    - was_bookmarked: 북마크 여부
    - dwell_time_ms: 상세 페이지 체류 시간 (밀리초)
    """
    was_clicked: bool = False
    was_bookmarked: bool = False
    dwell_time_ms: Optional[int] = None


def compute_reward(sig: InteractionSignal) -> float:
    """
    InteractionSignal을 바탕으로 단일 scalar reward를 계산.

    기본 정책 (예시):
      - 클릭: +1.0
      - 북마크: +1.0 (클릭과 별도로 가산)
      - dwell_time_ms:
          60초 이상: +0.5
          30초 이상: +0.25

    NOTE:
      - 이 함수는 쉽게 수정/실험할 수 있도록 별도 모듈로 분리했다.
      - 나중에 프로젝트 상황에 맞게 가중치만 바꿔도 된다.
    """
    reward = 0.0

    if sig.was_clicked:
        reward += 1.0

    if sig.was_bookmarked:
        reward += 1.0

    if sig.dwell_time_ms is not None:
        if sig.dwell_time_ms >= 60_000:
            reward += 0.5
        elif sig.dwell_time_ms >= 30_000:
            reward += 0.25

    return reward