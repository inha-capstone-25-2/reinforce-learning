from typing import Dict, Any, List

from .recommend import recommend_user, recommend_similar_papers


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