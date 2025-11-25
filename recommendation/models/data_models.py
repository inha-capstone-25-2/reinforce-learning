from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Paper:
    mongo_id: str
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    update_date: Optional[datetime] = None
    bookmark_count: int = 0
    view_count: int = 0
    difficulty_level: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    embedding_vector: Optional[List[float]] = None


@dataclass
class UserProfile:
    user_id: int
    interests_keywords: List[str] = field(default_factory=list)
    interests_categories: List[str] = field(default_factory=list)
    bookmarked_paper_ids: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    explicit_categories: Optional[List[str]] = None



@dataclass
class RecommendationResult:
    paper: Paper
    score: float
    features: Dict[str, float]

    def to_frontend_dict(self) -> Dict[str, Any]:
        """
        프론트엔드 요구사항에 맞춘 응답 포맷으로 변환.

        - id: 논문 ID (arxiv_id 우선, 없으면 mongo_id)
        - title, authors, abstract, categories
        - summary: 요약(영어/한글 중 하나, 실제 구현에 따라 조정 가능)
        - externalUrl: arXiv 원문 링크 (가능한 경우)
        - score: 추천 점수
        - features: 서브 스코어(카테고리, 키워드, 인기도, 최신성 등)
        """
        paper_id = self.paper.arxiv_id or self.paper.mongo_id
        return {
            "id": paper_id,
            "title": self.paper.title,
            "authors": self.paper.authors,
            "abstract": self.paper.abstract,
            "categories": self.paper.categories,
            # summary 구조는 프로젝트 사양에 맞게 조정 가능
            "summary": self.paper.summary,
            "externalUrl": f"https://arxiv.org/abs/{paper_id}" if self.paper.arxiv_id else None,
            "score": self.score,
            "features": self.features,
        }