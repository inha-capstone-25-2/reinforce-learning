from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Paper:
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    keywords: List[str]
    embedding_vector: List[float]
    publication_date: str
    journal_conference: str
    doi: str
    citation_count: int
    view_count: int
    click_count: int
    like_count: int
    difficulty_level: str  # beginner, intermediate, advanced

@dataclass
class UserViewHistory:
    paper_id: str
    timestamp: str
    dwell_time: int

@dataclass
class UserClickHistory:
    paper_id: str
    timestamp: str

@dataclass
class UserSearchHistory:
    query: str
    timestamp: str
    result_count: int

@dataclass
class User:
    user_id: str
    interests: List[str]
    research_field: str
    view_history: List[UserViewHistory]
    click_history: List[UserClickHistory]
    search_history: List[UserSearchHistory]
    bookmarks: List[str]
    preferred_categories: Dict[str, float]

@dataclass
class Interaction:
    interaction_id: str
    user_id: str
    paper_id: str
    session_id: str
    action_type: str  # view, click, like, bookmark, share
    timestamp: str
    position: int
    dwell_time: int
    scroll_depth: float
    recommendation_model: str