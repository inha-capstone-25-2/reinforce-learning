from typing import Dict, Any, List

from .recommend import (
    recommend_user,
    recommend_similar_papers,
    recommend_user_hybrid,
)


def get_user_recommendations(user_id: int, limit: int = 6) -> Dict[str, Any]:
    results = recommend_user(user_id, top_k=limit)
    return {
        "user_id": user_id,
        "count": len(results),
        "results": results,
    }


def get_similar_paper_recommendations(paper_id: str, limit: int = 6) -> Dict[str, Any]:
    results = recommend_similar_papers(paper_id, top_k=limit)
    return {
        "paper_id": paper_id,
        "count": len(results),
        "results": results,
    }


def get_user_recommendations_rl(user_id: int, limit: int = 6, candidate_k: int = 100) -> Dict[str, Any]:
    """
    RL (Contextual Bandit) reranking 기반 추천 API.
    백엔드는 이 함수를 import해서 /recommendations/rl 엔드포인트에서 호출하면 된다.
    """
    results = recommend_user_hybrid(user_id, top_k=limit, candidate_k=candidate_k)
    return {
        "user_id": user_id,
        "count": len(results),
        "results": results,
        "mode": "rule_based+rl",
    }