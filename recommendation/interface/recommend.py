from typing import List, Dict, Any, Optional

from ..data.data_loader import MongoDataLoader
from ..rule_based.rule_based_recommender import RuleBasedRecommender
from ..models.data_models import RecommendationResult
from ..service.pipeline import recommend_for_user_hybrid

_recommender: Optional[RuleBasedRecommender] = None


def _get_recommender() -> RuleBasedRecommender:
    global _recommender
    if _recommender is None:
        loader = MongoDataLoader()
        _recommender = RuleBasedRecommender(loader)
    return _recommender


def recommend_user(user_id: int, top_k: int = 6) -> List[Dict[str, Any]]:
    rec = _get_recommender().recommend_for_user(user_id, top_k)
    return [r.to_frontend_dict() for r in rec]

def recommend_similar_papers(paper_id: str, top_k: int = 6):
    rec = _get_recommender().recommend_similar_papers(paper_id, top_k)
    return [r.to_frontend_dict() for r in rec]


def recommend_user_hybrid(user_id: int, top_k: int = 6, candidate_k: int = 100, base_paper_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Rule-based 100개 → RL reranking → 최종 top_k 개.
    프론트/백엔드가 RL 기반 추천을 쓰고 싶을 때 이 함수를 호출하면 된다.
    """
    recs: List[RecommendationResult] = recommend_for_user_hybrid(
        user_id=user_id,
        top_k=top_k,
        candidate_k=candidate_k,
        base_paper_id=base_paper_id,
    )
    return [r.to_frontend_dict() for r in recs]