from typing import List, Optional

from ..data.data_loader import MongoDataLoader
from ..models.data_models import UserProfile, RecommendationResult, Paper
from .scoring import compute_total_score


class RuleBasedRecommender:
    def __init__(self, data_loader: MongoDataLoader):
        self.data_loader = data_loader


    def _similarity_bonus(self, paper: Paper, base: Paper) -> float:
        bonus = 0.0

        # 1) category overlap (0.2)
        if base.categories:
            cat_overlap = len(set(paper.categories) & set(base.categories))
            if cat_overlap > 0:
                bonus += 0.2

        # 2) keyword overlap (0.2)
        if base.keywords:
            kw_overlap = len(set(paper.keywords) & set(base.keywords))
            if kw_overlap > 0:
                bonus += 0.2

        return bonus
    
    # 수정됨. 유저 + 논문 유사도
    def recommend_for_user(
            self,
            user_id: int,
            top_k: int = 6,
            base_paper_id: Optional[str] = None
        ) -> List[RecommendationResult]:

        profile: UserProfile = self.data_loader.build_user_profile(user_id)

        # base 논문 로딩
        base_paper = None
        if base_paper_id:
            base_paper = self.data_loader.get_paper_by_arxiv_id(base_paper_id)

        candidates = self.data_loader.get_candidate_papers_for_user(profile)
        results = []

        for p in candidates:
            score, feats = compute_total_score(p, profile)

            # base 논문 유사도 추가
            if base_paper:
                sim = self._similarity_bonus(p, base_paper)
                score += sim
                feats["similarity_bonus"] = sim

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