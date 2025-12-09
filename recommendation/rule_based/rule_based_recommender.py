from typing import List, Optional

from ..data.data_loader import MongoDataLoader
from ..models.data_models import UserProfile, RecommendationResult, Paper
from .scoring import compute_total_score


class RuleBasedRecommender:
    def __init__(self, data_loader: MongoDataLoader):
        self.data_loader = data_loader


    def _similarity_bonus(self, paper: Paper, base: Paper) -> float:
        bonus = 0.0

        # Category similarity
        cats1 = set(paper.categories or [])
        cats2 = set(base.categories or [])
        cat_overlap = len(cats1 & cats2)

        if cat_overlap >= 3:
            bonus += 0.8
        elif cat_overlap == 2:
            bonus += 0.5
        elif cat_overlap == 1:
            bonus += 0.3
        else:
            bonus += 0.05  # 완전 다른 분야라도 최소점수 => 0 안되게 


        # Keyword similarity
        kw1 = set(paper.keywords or [])
        kw2 = set(base.keywords or [])
        kw_overlap = len(kw1 & kw2)

        if kw_overlap >= 3:
            bonus += 0.5
        elif kw_overlap == 2:
            bonus += 0.3
        elif kw_overlap == 1:
            bonus += 0.2
        else:
            bonus += 0.03

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
        # 자기 자신은 추천하지 않게 중복 피하기.
        if base_paper_id:
            candidates = [p for p in candidates if p.arxiv_id != base_paper_id]
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