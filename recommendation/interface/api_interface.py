from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional

from ..data.data_loader import MongoDataLoader
from ..models.data_models import RecommendationResult
from .recommend import recommend_user, recommend_user_hybrid, recommend_similar_papers
from ..rl.reward import compute_reward

logger = logging.getLogger(__name__)

# MongoDataLoader / Recommender 싱글톤
_loader_singleton: Optional[MongoDataLoader] = None


def _get_loader() -> MongoDataLoader:
    global _loader_singleton
    if _loader_singleton is None:
        _loader_singleton = MongoDataLoader()
    return _loader_singleton



# ------------------------------------------------------
# 룰베이스 추천 API + 노출 로그 기록
# ------------------------------------------------------
def get_user_recommendations(
    user_id: int,
    limit: int = 10,
    log_exposure: bool = True,
    request_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    
    #룰 베이스 추천 (기본 추천 API)
    recs = recommend_user(user_id, top_k=limit)
    results = recs

    loader = _get_loader()
    recommendation_id: Optional[str] = None

    if log_exposure:
        recommendation_id = loader.log_recommendation_event(
            user_id=user_id,
            results=results,
            mode="rule_based",
            request_meta=request_meta,
        )

    return {
        "user_id": user_id,
        "count": len(results),
        "results": results,
        "mode": "rule_based",
        "recommendation_id": recommendation_id,
    }


# ------------------------------------------------------
# ① -2 유사 논문 추천 API
# ------------------------------------------------------
def get_similar_paper_recommendations(paper_id: str, limit: int = 6) -> Dict[str, Any]:
    """
    특정 논문과 유사한 논문 추천.
    """
    results = recommend_similar_papers(paper_id, top_k=limit)
    return {
        "paper_id": paper_id,
        "count": len(results),
        "results": results,
    }


# ------------------------------------------------------
# ② 룰베이스 + RL 하이브리드 추천 API + 노출 로그 기록

# ------------------------------------------------------
def get_user_recommendations_rl(
    user_id: int,
    limit: int = 10,
    candidate_k: int = 200,
    base_paper_id: Optional[str] = None,
    log_exposure: bool = True,
    request_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    
    #룰 + RL 하이브리드 추천 API
    recs= recommend_user_hybrid(
        user_id=user_id,
        top_k=limit,
        candidate_k=candidate_k,
        base_paper_id=base_paper_id,
    )
    results = recs

    loader = _get_loader()
    recommendation_id: Optional[str] = None

    if log_exposure:
        recommendation_id = loader.log_recommendation_event(
            user_id=user_id,
            results=results,
            mode="rule_based+rl",
            request_meta=request_meta,
        )

    return {
        "user_id": user_id,
        "count": len(results),
        "results": results,
        "mode": "rule_based+rl",
        "recommendation_id": recommendation_id,
    }


# ------------------------------------------------------
# 클릭 / 북마크 등의 상호작용 로그 API→ 프론트에서 별도 엔드포인트로 호출
# ------------------------------------------------------
def log_recommendation_interaction(
    user_id: int,
    paper_id: str,
    action_type: str = "click",
    recommendation_id: Optional[str] = None,
    position: Optional[int] = None,
    dwell_time: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    프론트에서 클릭/북마크 발생 시 백엔드에서 호출할 함수.

    예) FastAPI 기준:
        @post("/recommendations/interactions")
        def log_interaction(req: InteractionRequest):
            return log_recommendation_interaction(
                user_id=req.user_id,
                paper_id=req.paper_id,
                action_type=req.action_type,
                recommendation_id=req.recommendation_id,
                position=req.position,
                dwell_time=req.dwell_time,
                meta=req.meta,
            )
    """
    logger.info("=" * 60)
    logger.info("[RL Interaction] 📥 상호작용 로그 수신")
    logger.info(f"[RL Interaction] 👤 user_id={user_id}, 📄 paper_id={paper_id}")
    logger.info(f"[RL Interaction] 🎯 action_type={action_type}, position={position}, dwell_time={dwell_time}")
    
    loader = _get_loader()

    # reward 계산에 필요한 최소 필드만 모아 전달
    interaction_payload = {
        "action_type": action_type,
        "dwell_time": dwell_time,
        "meta": meta or {},
    }
    reward = compute_reward(interaction_payload)
    
    # reward 계산 상세 로그
    logger.info(f"[RL Interaction] 💰 Reward 계산 완료: {reward:.2f}")
    if action_type == "click":
        logger.info(f"[RL Interaction]   └─ click: +1.0")
    elif action_type == "bookmark":
        logger.info(f"[RL Interaction]   └─ bookmark: +3.0")
    if dwell_time is not None:
        if dwell_time >= 3.0:
            logger.info(f"[RL Interaction]   └─ dwell_time >= 3초: +0.3")
        elif dwell_time <= 1.0:
            logger.info(f"[RL Interaction]   └─ dwell_time <= 1초 (이탈): -0.2")

    interaction_id = loader.log_interaction(
        user_id=user_id,
        paper_id=paper_id,
        action_type=action_type,
        recommendation_id=recommendation_id,
        position=position,
        dwell_time=dwell_time,
        reward=reward,
        meta=meta,
    )

    logger.info(f"[RL Interaction] ✅ MongoDB 저장 완료: interaction_id={interaction_id}")
    logger.info(f"[RL Interaction] 🎁 최종 reward: {reward}")
    logger.info("=" * 60)

    return {
        "ok": True,
        "interaction_id": interaction_id,
        "reward": reward,
    }

