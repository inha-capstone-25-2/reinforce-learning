from typing import List, Dict, Any, Optional

from ..data.data_loader import MongoDataLoader
from ..rule_based.rule_based_recommender import RuleBasedRecommender
from ..models.data_models import RecommendationResult


_recommender: Optional[RuleBasedRecommender] = None


def _get_recommender() -> RuleBasedRecommender:
    global _recommender
    if _recommender is None:
        loader = MongoDataLoader()
        _recommender = RuleBasedRecommender(loader)
    return _recommender


def recommend_user(user_id: int, top_k: int = 6) -> List[Dict[str, Any]]:
    rec = _get_recommender().recommend_for_user(user_id, top_k)
    return [r.to_dict() for r in rec]


def recommend_similar_papers(paper_id: str, top_k: int = 6):
    rec = _get_recommender().recommend_similar_papers(paper_id, top_k)
    return [r.to_dict() for r in rec]