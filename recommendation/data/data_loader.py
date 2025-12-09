from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any, Sequence
from uuid import uuid4

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING
from sshtunnel import SSHTunnelForwarder

from ..models.data_models import Paper, UserProfile
from .preprocess import tokenize_keywords


# -----------------------------------------
#  환경변수 로드 (backend-secret/dev/.env)
# -----------------------------------------
_CURRENT_DIR = Path(__file__).resolve().parent
# data -> recommendation -> reinforce-learning
_PROJECT_ROOT = _CURRENT_DIR.parent.parent  # reinforce-learning 폴더
_ENV_PATH = _PROJECT_ROOT / "backend-secret" / "dev" / ".env"
load_dotenv(_ENV_PATH)

# -----------------------------------------
#  SSH 터널링 설정
# -----------------------------------------
# PEM 키 경로 (backend-secret 폴더의 capstone-02.pem)
SSH_PEM_KEY_PATH = _PROJECT_ROOT / "backend-secret" / "capstone-02.pem"

# SSH 서버 정보 (MongoDB 서버의 public IP)
SSH_HOST = os.getenv("MONGO_PUBLIC_IP")
SSH_PORT = 22
SSH_USERNAME = "ubuntu"

# MongoDB 서버 정보 (private IP - SSH 터널 내부에서 접근)
MONGODB_PRIVATE_IP = os.getenv("MONGO_HOST")
MONGODB_PORT = int(os.getenv("MONGO_PORT", "27017"))
MONGODB_USERNAME = os.getenv("MONGO_USER")
MONGODB_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGODB_DB_NAME = os.getenv("MONGO_DB", "arxiv")

# 전역 SSH 터널
_ssh_tunnel: Optional[SSHTunnelForwarder] = None


def get_ssh_tunnel() -> SSHTunnelForwarder:
    #SSH 터널을 가져오거나 생성
    import paramiko

    global _ssh_tunnel
    if _ssh_tunnel is None or not _ssh_tunnel.is_active:
        # PEM 키 파일에서 RSA 키 로드
        pkey = paramiko.RSAKey.from_private_key_file(str(SSH_PEM_KEY_PATH))

        _ssh_tunnel = SSHTunnelForwarder(
            (SSH_HOST, SSH_PORT),
            ssh_username=SSH_USERNAME,
            ssh_pkey=pkey,
            # MongoDB는 EC2 내부에서 127.0.0.1:27017로 리스닝
            remote_bind_address=("127.0.0.1", MONGODB_PORT),
            local_bind_address=("127.0.0.1", 0),  # 자동으로 사용 가능한 포트 할당
            allow_agent=False,  # SSH agent 사용 안함
            host_pkey_directories=[],  # 호스트 키 디렉토리 비활성화
        )
        _ssh_tunnel.start()
    return _ssh_tunnel


class MongoDataLoader:
    """
    MongoDB 기반 UserProfile + Paper 로딩 + 로그 기록 클래스
    """

    def __init__(self, client: Optional[MongoClient] = None, db_name: str = None):
        # -----------------------------
        # MongoDB 연결
        # -----------------------------
        if client is None:
            # SSH 터널 설정
            tunnel = get_ssh_tunnel()
            local_port = tunnel.local_bind_port

            auth_source = os.getenv("MONGO_AUTH_SOURCE", "admin")
            uri = (
                f"mongodb://{MONGODB_USERNAME}:{MONGODB_PASSWORD}"
                f"@127.0.0.1:{local_port}/?authSource={auth_source}&directConnection=true"
            )
            client = MongoClient(
                uri,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
            )

        self.client = client
        self.db = self.client[db_name or MONGODB_DB_NAME]

        # 컬렉션
        self.col_papers = self.db["papers"]
        self.col_bookmarks = self.db["bookmarks"]
        self.col_search_history = self.db["search_history"]
        self.col_user_activities = self.db["user_activities"]
        # + 추천 로그 컬렉션
        self.col_reco_events = self.db["recommendation_events"]
        self.col_reco_interactions = self.db["recommendation_interactions"]

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
            arxiv_id=doc.get("_id"),
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
        # papers 컬렉션에서 _id = "0704.0001" 형식이므로 _id로 조회
        doc = self.col_papers.find_one({"_id": arxiv_id})
        return self._doc_to_paper(doc) if doc else None

    def get_recent_papers(self, limit: int = 200):
        cursor = self.col_papers.find().sort("update_date", DESCENDING).limit(limit)
        return [self._doc_to_paper(d) for d in cursor]

    def get_papers_by_categories(self, categories: Iterable[str], limit=300):
        cursor = (
            self.col_papers.find({"categories": {"$in": list(categories)}})
            .sort("update_date", DESCENDING)
            .limit(limit)
        )
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
        bookmarked_papers = [self.get_paper_by_arxiv_id(pid) for pid in bookmarked_ids]
        bookmarked_papers = [p for p in bookmarked_papers if p]

        categories: List[str] = []
        keywords: List[str] = []

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
    def get_candidate_papers_for_user(
        self, profile: UserProfile, limit_per_source: int = 200
    ):
        candidates: Dict[str, Paper] = {}

        # 관심 카테고리 기반
        if profile.interests_categories:
            for p in self.get_papers_by_categories(
                profile.interests_categories, limit_per_source
            ):
                if p.arxiv_id:
                    candidates[p.arxiv_id] = p

        # 최신 기반 추가
        for p in self.get_recent_papers(limit_per_source):
            if p.arxiv_id:
                candidates.setdefault(p.arxiv_id, p)

        # 이미 북마크한 논문 제외
        for pid in profile.bookmarked_paper_ids:
            candidates.pop(pid, None)

        return list(candidates.values())

    # ------------------------------------------------------
    # 추천 노출 로그 저장
    # ------------------------------------------------------
    def log_recommendation_event(
        self,
        user_id: int,
        results: Sequence[Dict[str, Any]],
        mode: str,
        request_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        추천 결과 노출 시 1회 호출.
        - results: get_user_recommendations* 가 반환하는 results 리스트 형태 그대로 사용
        """
        recommendation_id = str(uuid4())
        now = datetime.utcnow()

        items = []
        for idx, r in enumerate(results):
            items.append(
                {
                    "paper_id": r.get("id"),
                    "position": idx,
                    "score": float(r.get("score", 0.0)),
                    "features": r.get("features", {}),
                }
            )

        doc = {
            "_id": recommendation_id,
            "recommendation_id": recommendation_id,
            "user_id": user_id,
            "mode": mode,  # "rule_based" / "rule_based+rl"
            "items": items,
            "request_meta": request_meta or {},
            "created_at": now,
        }
        self.col_reco_events.insert_one(doc)
        return recommendation_id

    # ------------------------------------------------------
    # 클릭/북마크 등 상호작용 로그 저장
    # ------------------------------------------------------
    def log_interaction(
        self,
        user_id: int,
        paper_id: str,
        action_type: str = "click",
        recommendation_id: Optional[str] = None,
        position: Optional[int] = None,
        dwell_time: Optional[float] = None,
        reward: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        
        #사용자의 클릭/북마크/닫기 등 상호작용을 기록.
        
        interaction_id = str(uuid4())
        now = datetime.utcnow()

        doc = {
            "_id": interaction_id,
            "interaction_id": interaction_id,
            "user_id": user_id,
            "paper_id": paper_id,
            "action_type": action_type,  # "click", "bookmark", "close" 등
            "recommendation_id": recommendation_id,
            "position": position,
            "dwell_time": dwell_time,
            "reward": reward,
            "meta": meta or {},
            "created_at": now,
        }
        self.col_reco_interactions.insert_one(doc)
        return interaction_id
