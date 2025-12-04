from typing import List

from ..data.data_loader import MongoDataLoader
from ..models.data_models import UserProfile, RecommendationResult
from .scoring import compute_total_score


class RuleBasedRecommender:
    def __init__(self, data_loader: MongoDataLoader):
        self.data_loader = data_loader

    def recommend_for_user(self, user_id: int, top_k: int = 6) -> List[RecommendationResult]:
        profile: UserProfile = self.data_loader.build_user_profile(user_id)
        candidates = self.data_loader.get_candidate_papers_for_user(profile)

        results = []
        for p in candidates:
            score, feats = compute_total_score(p, profile)
            results.append(RecommendationResult(p, score, feats))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def recommend_similar_papers(self, paper_id: str, top_k: int = 6):
        base = self.data_loader.get_paper_by_arxiv_id(paper_id)
        if not base:
            return []

        profile = UserProfile(
            user_id=-1,
            interests_keywords=base.keywords,
            interests_categories=base.categories,
        )

        candidates = self.data_loader.get_papers_by_categories(base.categories, limit=300)
        candidates = [p for p in candidates if p.arxiv_id != base.arxiv_id]

        results = []
        for p in candidates:
            score, feats = compute_total_score(p, profile)
            results.append(RecommendationResult(p, score, feats))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]