from __future__ import annotations
import os
from datetime import datetime
from typing import Iterable, List, Optional, Dict, Any

from bson import ObjectId
from pymongo import MongoClient, DESCENDING

from ..models.data_models import Paper, UserProfile
from .preprocess import tokenize_keywords


# -----------------------------------------
#  SSH 터널링을 사용하므로 host=localhost 로 접속됨
# -----------------------------------------
MONGODB_HOST = "localhost"
MONGODB_PORT = 27017
MONGODB_USERNAME = "rsrs-root"
MONGODB_PASSWORD = "KIQu3jebjHNhTEE6mm5tgj2oNjYr7J805k2JLbE0AVo"
MONGODB_DB_NAME = "arxiv"


class MongoDataLoader:
    """
    MongoDB 기반 UserProfile + Paper 로딩 클래스
    """

    def __init__(self, client: Optional[MongoClient] = None, db_name: str = None):
        # -----------------------------
        # 실제 MongoDB 연결 (SSH 포트포워딩 기반)
        # -----------------------------
        if client is None:
            uri = f"mongodb://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_HOST}:{MONGODB_PORT}"
            client = MongoClient(uri)

        self.client = client
        self.db = self.client[db_name or MONGODB_DB_NAME]

        # Collections
        self.col_papers = self.db["papers"]
        self.col_bookmarks = self.db["bookmarks"]
        self.col_search_history = self.db["search_history"]
        self.col_user_activities = self.db["user_activities"]

    # ------------------------------------------------------
    # Paper Document → Paper dataclass 변환
    # ------------------------------------------------------
    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except Exception:
                return None
        return None

    @staticmethod
    def _doc_to_paper(doc: Dict[str, Any]) -> Paper:
        return Paper(
            mongo_id=str(doc["_id"]),
            arxiv_id=doc.get("id"),  # ★ 실제 DB는 arXiv ID 문자열
            title=doc.get("title"),
            abstract=doc.get("abstract"),
            authors=doc.get("authors"),
            categories=list(doc.get("categories") or []),
            keywords=list(doc.get("keywords") or []),
            update_date=MongoDataLoader._parse_datetime(doc.get("update_date")),
            bookmark_count=int(doc.get("bookmark_count") or 0),
            view_count=int(doc.get("view_count") or 0),
            difficulty_level=doc.get("difficulty_level"),
            summary=doc.get("summary"),
            embedding_vector=doc.get("embedding_vector"),
        )

    # ------------------------------------------------------
    # PAPER 조회 관련
    # ------------------------------------------------------
    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[Paper]:
        doc = self.col_papers.find_one({"id": arxiv_id})
        return self._doc_to_paper(doc) if doc else None

    def get_recent_papers(self, limit: int = 200):
        cursor = self.col_papers.find().sort("update_date", DESCENDING).limit(limit)
        return [self._doc_to_paper(d) for d in cursor]

    def get_papers_by_categories(self, categories: Iterable[str], limit=300):
        cursor = self.col_papers.find({"categories": {"$in": list(categories)}})\
                                .sort("update_date", DESCENDING)\
                                .limit(limit)
        return [self._doc_to_paper(d) for d in cursor]

    # ------------------------------------------------------
    # USER DATA 조회
    # ------------------------------------------------------
    def get_user_bookmarked_paper_ids(self, user_id: int) -> List[str]:
        cursor = self.col_bookmarks.find({"users_id": user_id})
        result = []
        for d in cursor:
            pid = d.get("paper_id")
            if isinstance(pid, str):
                result.append(pid)
        return result

    def get_user_search_queries(self, user_id: int, limit: int = 20):
        cursor = (
            self.col_search_history.find({"users_id": user_id})
            .sort("searched_at", DESCENDING)
            .limit(limit)
        )
        return [d.get("query") for d in cursor if d.get("query")]

    # ------------------------------------------------------
    # USER PROFILE 구성
    # ------------------------------------------------------
    def build_user_profile(self, user_id: int) -> UserProfile:
        # 북마크 기반
        bookmarked_ids = self.get_user_bookmarked_paper_ids(user_id)
        bookmarked_papers = [
            self.get_paper_by_arxiv_id(pid) for pid in bookmarked_ids
        ]
        bookmarked_papers = [p for p in bookmarked_papers if p]

        categories = []
        keywords = []

        for p in bookmarked_papers:
            categories.extend(p.categories)
            keywords.extend(p.keywords)

        # 검색 기반 키워드
        search_queries = self.get_user_search_queries(user_id)
        for q in search_queries:
            keywords.extend(tokenize_keywords(q))

        return UserProfile(
            user_id=user_id,
            interests_categories=sorted(set(categories)),
            interests_keywords=sorted(set(keywords)),
            bookmarked_paper_ids=bookmarked_ids,
            search_queries=search_queries,
            explicit_categories=None,
        )

    # ------------------------------------------------------
    # Candidate 생성
    # ------------------------------------------------------
    def get_candidate_papers_for_user(self, profile: UserProfile, limit_per_source=200):
        candidates = {}

        # 관심 카테고리 기반
        if profile.interests_categories:
            for p in self.get_papers_by_categories(profile.interests_categories, limit_per_source):
                candidates[p.arxiv_id] = p

        # 최신 기반 추가
        for p in self.get_recent_papers(limit_per_source):
            candidates.setdefault(p.arxiv_id, p)

        # 이미 북마크한 논문 제외
        for pid in profile.bookmarked_paper_ids:
            candidates.pop(pid, None)

        return list(candidates.values())
