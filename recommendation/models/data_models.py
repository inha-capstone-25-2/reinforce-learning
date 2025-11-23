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


@dataclass
class RecommendationResult:
    paper: Paper
    score: float
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """프론트 요구사항 기반 JSON 포맷"""

        paper = self.paper

        # year
        year = paper.update_date.year if paper.update_date else None

        # translated summary
        translated_summary = None
        if isinstance(paper.summary, dict):
            translated_summary = (
                paper.summary.get("ko")
                or paper.summary.get("translated")
                or paper.summary.get("kr")
            )

        # external URL
        external_url = None
        if paper.arxiv_id:
            external_url = f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"

        # summary 또는 abstract 일부
        summary_text = None
        if isinstance(paper.summary, dict):
            summary_text = (
                paper.summary.get("en")
                or paper.summary.get("summary")
                or paper.summary.get("abstract")
            )
        if not summary_text:
            summary_text = (paper.abstract[:300] + "...") if paper.abstract else None

        return {
            "id": paper.arxiv_id,
            "mongo_id": paper.mongo_id,
            "title": paper.title,
            "authors": paper.authors,
            "year": year,
            "publisher": "",
            "categories": paper.categories,
            "abstract": paper.abstract,
            "translatedSummary": translated_summary,
            "keywords": paper.keywords,
            "externalUrl": external_url,

            # 추천용
            "summary": summary_text,
            "score": self.score,
            "features": self.features,
        }