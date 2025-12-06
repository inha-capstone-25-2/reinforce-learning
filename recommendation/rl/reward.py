from __future__ import annotations
from typing import Mapping, Any, Optional

"""
reward.py

추천 상호작용 로그(recommendation_interactions)에 대해
보상 값을 계산하는 모듈.

현재 설계:
- click  : +1.0
- bookmark : +3.0
- dwell_time >= 3초 : +0.3
- dwell_time <= 1초 : -0.2   (바로 이탈한 경우 페널티)
"""


# 보상 하이퍼파라미터 (필요시 나중에 조정 가능)
CLICK_REWARD = 1.0
BOOKMARK_REWARD = 3.0
DWELL_THRESHOLD = 3.0  # seconds
DWELL_REWARD = 0.3
BOUNCE_THRESHOLD = 1.0  # seconds
BOUNCE_PENALTY = -0.2


def compute_reward(interaction: Mapping[str, Any]) -> float:
    """
    interaction dict 예시:
    {
        "action_type": "click" | "bookmark" | "close" | ...,
        "dwell_time": 2.5,   # seconds (없으면 None)
        "meta": {...}
    }
    """
    action_type: str = interaction.get("action_type", "")
    dwell_time: Optional[float] = interaction.get("dwell_time")

    reward = 0.0

    # 1) 기본 action 기반 보상
    if action_type == "click":
        reward += CLICK_REWARD
    elif action_type == "bookmark":
        reward += BOOKMARK_REWARD

    # 2) 체류시간 기반 보상/패널티
    if dwell_time is not None:
        if dwell_time >= DWELL_THRESHOLD:
            reward += DWELL_REWARD
        elif dwell_time <= BOUNCE_THRESHOLD:
            reward += BOUNCE_PENALTY

    return reward
