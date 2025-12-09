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
    """
    기존 recommend_for_user를 호출하던 부분을
    단일 recommend() 함수만 호출하도록 수정.
    """
    rec = _get_recommender().recommend(
        user_id=user_id,
        paper_id=None,# 유저 기반 추천
        top_k=top_k,
        candidate_k=top_k # 굳이 많은 후보 필요 없음
    )
    return [r.to_frontend_dict() for r in rec]


def recommend_similar_papers(
    paper_id: str,
    top_k: int = 6,
    user_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    기존 recommend_similar_papers도 단일 recommend() 기반으로 통합.
    user_id가 있으면 유저 관심도 반영 + 상세페이지 논문 기반
    user_id 없으면 순수 논문 기반 유사 논문 추천
    """
    rec = _get_recommender().recommend(
        user_id=user_id,      
        paper_id=paper_id,   
        top_k=top_k,
        candidate_k=300
    )
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