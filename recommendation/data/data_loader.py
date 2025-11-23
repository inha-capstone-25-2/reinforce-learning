from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, List, Optional, Dict, Any

from bson import ObjectId
from pymongo import MongoClient, DESCENDING

from ..models.data_models import Paper, UserProfile
from .preprocess import tokenize_keywords


MONGODB_HOST = "35.87.92.19"
MONGODB_PORT = 27017
MONGODB_USERNAME = "rsrs-root"
MONGODB_PASSWORD = "KIQu3jebjHNhTEE6mm5tgj2oNjYr7J805k2JLbE0AVo"
MONGODB_DB_NAME = "arxiv"


class MongoDataLoader:
    def __init__(self, client: Optional[MongoClient] = None, db_name: str = None):
        if client is None:
            if MONGODB_USERNAME and MONGODB_PASSWORD:
                uri = f"mongodb://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_HOST}:{MONGODB_PORT}"
            else:
                uri = f"mongodb://{MONGODB_HOST}:{MONGODB_PORT}"
            client = MongoClient(uri)

        self.client = client
        self.db = self.client[db_name or MONGODB_DB_NAME]

        self.col_papers = self.db["papers"]
        self.col_bookmarks = self.db["bookmarks"]
        self.col_search_history = self.db["search_history"]
        self.col_user_activities = self.db["user_activities"]

    # -------------- Paper Parsing -----------------

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except:
                return None
        return None

    @staticmethod
    def _doc_to_paper(doc: Dict[str, Any]) -> Paper:
        return Paper(
            mongo_id=str(doc["_id"]),
            arxiv_id=doc.get("id") or doc.get("arvix_id"),
            title=doc.get("title"),
            abstract=doc.get("abstract"),
            authors=doc.get("authors"),
            categories=list(doc.get("categories") or []),
            keywords=list(doc.get("keywords") or []),
            update_date=MongoDataLoader._parse_datetime(
                doc.get("update_date") or doc.get("published_date")
            ),
            bookmark_count=int(doc.get("bookmark_count") or 0),
            view_count=int(doc.get("view_count") or 0),
            difficulty_level=doc.get("difficulty_level"),
            summary=doc.get("summary"),
            embedding_vector=doc.get("embedding_vector"),
        )

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[Paper]:
        doc = self.col_papers.find_one({"id": arxiv_id}) or \
              self.col_papers.find_one({"arvix_id": arxiv_id})
        return self._doc_to_paper(doc) if doc else None

    def get_papers_by_mongo_ids(self, mongo_ids: Iterable[str]) -> List[Paper]:
        oids = [ObjectId(mid) for mid in mongo_ids if ObjectId.is_valid(mid)]
        if not oids:
            return []
        cursor = self.col_papers.find({"_id": {"$in": oids}})
        return [self._doc_to_paper(d) for d in cursor]

    def get_recent_papers(self, limit: int = 200) -> List[Paper]:
        cursor = self.col_papers.find().sort(
            [("update_date", DESCENDING)]
        ).limit(limit)
        return [self._doc_to_paper(d) for d in cursor]

    def get_papers_by_categories(self, categories: Iterable[str], limit: int = 300):
        cursor = self.col_papers.find({"categories": {"$in": list(categories)}})\
            .sort([("update_date", DESCENDING)]).limit(limit)
        return [self._doc_to_paper(d) for d in cursor]

    # ----------- User profile aggregation ------------

    def get_user_bookmarked_paper_ids(self, user_id: int) -> List[str]:
        cursor = self.col_bookmarks.find({"users_id": user_id})
        return [str(d["paper_id"]) for d in cursor if isinstance(d.get("paper_id"), ObjectId)]

    def get_user_search_queries(self, user_id: int, limit: int = 20) -> List[str]:
        cursor = self.col_search_history.find({"users_id": user_id})\
            .sort("searched_at", DESCENDING).limit(limit)
        return [d.get("query") for d in cursor if d.get("query")]

    def build_user_profile(self, user_id: int) -> UserProfile:
        bookmarked_ids = self.get_user_bookmarked_paper_ids(user_id)
        bookmarked_papers = self.get_papers_by_mongo_ids(bookmarked_ids)

        cats, kws = [], []
        for p in bookmarked_papers:
            cats.extend(p.categories)
            kws.extend(p.keywords)

        search_queries = self.get_user_search_queries(user_id)
        for q in search_queries:
            kws.extend(tokenize_keywords(q))

        return UserProfile(
            user_id=user_id,
            interests_categories=sorted(set(cats)),
            interests_keywords=sorted(set(kws)),
            bookmarked_paper_ids=bookmarked_ids,
            search_queries=search_queries,
        )

    # ----------- Candidate selection ------------

    def get_candidate_papers_for_user(self, profile: UserProfile, limit_per_source=200):
        candidates = {}

        if profile.interests_categories:
            for p in self.get_papers_by_categories(profile.interests_categories, limit_per_source):
                candidates[p.mongo_id] = p

        for p in self.get_recent_papers(limit_per_source):
            candidates.setdefault(p.mongo_id, p)

        for mid in profile.bookmarked_paper_ids:
            candidates.pop(mid, None)

        return list(candidates.values())