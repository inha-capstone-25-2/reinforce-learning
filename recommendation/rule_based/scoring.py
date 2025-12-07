from datetime import datetime
from math import log1p
from typing import Dict, Tuple, Set

from ..models.data_models import Paper, UserProfile
from ..data.preprocess import tokenize_keywords

# Weight Definitions
W_EXPLICIT_CAT = 0.50
W_BOOKMARK_CAT = 0.30
W_BOOKMARK_KW = 0.35
W_SEARCH_KW = 0.15
W_POPULARITY = 0.05 # 낮춤
W_RECENCY = 0.05 #낮춤


# Keyword Score 

def _keyword_score(paper: Paper, profile: UserProfile) -> float:
    
    paper_kw: Set[str] = set(k.lower() for k in paper.keywords)

    if paper.title:
        paper_kw.update(tokenize_keywords(paper.title))
    if paper.abstract:
        paper_kw.update(tokenize_keywords(paper.abstract))

    if not paper_kw:
        return 0.0

    bookmark_kw = set()
    search_kw = set()

    for p_kw in profile.bookmarked_paper_ids:
        # NOTE: keywords in Paper level parsed separately
        pass  # bookmark keywords already added via interests_keywords

    for q in profile.search_queries:
        search_kw.update(tokenize_keywords(q))

    all_profile_kw = set(profile.interests_keywords)
    bookmark_kw = all_profile_kw - search_kw

    overlap_bookmark = len(bookmark_kw & paper_kw) / len(bookmark_kw) if bookmark_kw else 0.0
    overlap_search = len(search_kw & paper_kw) / len(search_kw) if search_kw else 0.0

    # 가중치 합산
    score = (W_BOOKMARK_KW * overlap_bookmark +
             W_SEARCH_KW * overlap_search)

    return min(score, 1.0)


# Category Score 

def _category_score(paper: Paper, profile: UserProfile) -> float:
    paper_cats = set(paper.categories)

    explicit = set(profile.explicit_categories or [])
    all_cats = set(profile.interests_categories)

    bookmark_cats = all_cats - explicit if explicit else all_cats

    s_explicit = (len(explicit & paper_cats) / len(explicit)) if explicit else 0.0
    s_bookmark = (len(bookmark_cats & paper_cats) / len(bookmark_cats)) if bookmark_cats else 0.0

    return min(W_EXPLICIT_CAT * s_explicit + W_BOOKMARK_CAT * s_bookmark, 1.0)


# Popularity Score 

def _popularity_score(paper: Paper) -> float:
    b = max(paper.bookmark_count, 0)
    v = max(paper.view_count, 0)
    return (log1p(b) + log1p(v)) / 2.0


# Recency Score 

def _recency_score(paper: Paper, now: datetime) -> float:
    if not paper.update_date:
        return 0.0
    days = (now - paper.update_date).days
    return 0.5 ** (days / 730.0)  # half-life ~2 years


# Total Score

def compute_total_score(paper: Paper, profile: UserProfile,
                        now: datetime = None) -> Tuple[float, Dict[str, float]]:

    now = now or datetime.utcnow()

    s_kw = _keyword_score(paper, profile)
    s_cat = _category_score(paper, profile)
    s_pop = min(_popularity_score(paper) / 10.0, 1.0)
    s_rec = _recency_score(paper, now)

    total = (
        s_kw +
        s_cat +
        W_POPULARITY * s_pop +
        W_RECENCY * s_rec
    )

    return total, {
        "keyword": s_kw,
        "category": s_cat,
        "popularity": s_pop,
        "recency": s_rec,
    }
