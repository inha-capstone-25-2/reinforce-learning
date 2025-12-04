from __future__ import annotations

from typing import List

from ..data.data_loader import MongoDataLoader
from ..models.data_models import RecommendationResult
from ..rule_based.rule_based_recommender import RuleBasedRecommender
from .reranker import RLBanditReranker


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
) -> List[RecommendationResult]:
    """
    1) Rule-based로 candidate_k개 후보 생성
    2) RL(Contextual Bandit)으로 rerank
    3) 최종 top_k개 반환
    """
    rule_rec = _get_rule_recommender()
    rl_reranker = _get_rl_reranker()

    # 1) Rule-based 후보 100개
    candidates: List[RecommendationResult] = rule_rec.recommend_for_user(
        user_id=user_id,
        top_k=candidate_k,
    )

    if not candidates:
        return []

    # 2) RL로 rerank → 최종 6개
    final_results = rl_reranker.rerank(
        user_id=user_id,
        candidates=candidates,
        top_k=top_k,
    )
    return final_results
