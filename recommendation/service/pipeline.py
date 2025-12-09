from __future__ import annotations
import logging
from typing import List, Optional

from ..data.data_loader import MongoDataLoader
from ..models.data_models import RecommendationResult, UserProfile
from ..rule_based.rule_based_recommender import RuleBasedRecommender, compute_total_score
from .reranker import RLBanditReranker
from ..rule_based.rule_based_recommender import RuleBasedRecommender

logger = logging.getLogger(__name__)

_loader: MongoDataLoader | None = None
_rule_rec: RuleBasedRecommender | None = None
_rl_reranker: RLBanditReranker | None = None


def _get_loader() -> MongoDataLoader:
    global _loader
    if _loader is None:
        _loader = MongoDataLoader()
    return _loader


def _get_rule_recommender() -> RuleBasedRecommender:
    global _rule_rec
    if _rule_rec is None:
        _rule_rec = RuleBasedRecommender(_get_loader())
    return _rule_rec


def _get_rl_reranker() -> RLBanditReranker:
    global _rl_reranker
    if _rl_reranker is None:
        _rl_reranker = RLBanditReranker(_get_loader())
    return _rl_reranker


def recommend_for_user_hybrid(
    user_id: int,
    top_k: int = 6,
    candidate_k: int = 100,
    base_paper_id: Optional[str] = None,
) -> List[RecommendationResult]:
    """
    1) Rule-based로 candidate_k개 후보 생성
    2) RL(Contextual Bandit)으로 rerank
    3) 최종 top_k개 반환
    """
    logger.info("=" * 60)
    logger.info("[RL Pipeline] 🚀 Hybrid 추천 시작")
    logger.info(f"[RL Pipeline] 📋 Parameters: user_id={user_id}, top_k={top_k}, candidate_k={candidate_k}, base_paper_id={base_paper_id}")
    
    rule_rec = _get_rule_recommender()
    rl_reranker = _get_rl_reranker()

    # 1) Rule-based 후보 100개
    logger.info(f"[RL Pipeline] 📊 Step 1: Rule-based 후보 {candidate_k}개 생성 중...")
    candidates = rule_rec.recommend_for_user(
        user_id=user_id,
        top_k=candidate_k,
        base_paper_id=base_paper_id,
    )
    logger.info(f"[RL Pipeline] ✅ Rule-based 후보 {len(candidates)}개 생성 완료")

    if not candidates:
        logger.warning("[RL Pipeline] ⚠️ 후보가 없어 빈 결과 반환")
        return []

    # 후보 상위 3개 미리보기
    for i, c in enumerate(candidates[:3]):
        logger.info(f"[RL Pipeline]   후보 {i+1}: {c.paper.title[:50]}... (rule_score={c.score:.4f})")

    # 2) RL로 rerank → 최종 6개
    logger.info(f"[RL Pipeline] 🤖 Step 2: RL Bandit으로 reranking 중...")
    final_results = rl_reranker.rerank(
        user_id=user_id,
        candidates=candidates,
        top_k=top_k,
    )
    
    logger.info(f"[RL Pipeline] ✅ RL reranking 완료 → 최종 {len(final_results)}개 선택")
    
    # 최종 결과 로그
    for i, r in enumerate(final_results):
        rl_score = r.features.get("rl_score", "N/A")
        rule_score = r.features.get("rule_score", r.score)
        sim_bonus = r.features.get("similarity_bonus", 0)
        rl_score_str = f"{rl_score:.4f}" if isinstance(rl_score, (int, float)) else str(rl_score)
        rule_score_str = f"{rule_score:.4f}" if isinstance(rule_score, (int, float)) else str(rule_score)
        logger.info(f"[RL Pipeline]   결과 {i+1}: {r.paper.title[:40]}... | rl={rl_score_str} | rule={rule_score_str} | sim_bonus={sim_bonus}")
    
    logger.info("[RL Pipeline] 🎯 Hybrid 추천 완료")
    logger.info("=" * 60)
    
    return final_results

