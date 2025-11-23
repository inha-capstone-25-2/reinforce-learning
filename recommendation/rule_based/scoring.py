from datetime import datetime
from math import log1p
from typing import Dict, Tuple

from ..models.data_models import Paper, UserProfile
from ..data.preprocess import tokenize_keywords

WEIGHT_KEYWORD = 0.4
WEIGHT_CATEGORY = 0.3
WEIGHT_POPULARITY = 0.2
WEIGHT_RECENCY = 0.1


def _keyword_score(paper: Paper, profile: UserProfile) -> float:
    user_kw = set(k.lower() for k in profile.interests_keywords)
    paper_kw = set(k.lower() for k in paper.keywords)

    if paper.title:
        paper_kw.update(tokenize_keywords(paper.title))
    if paper.abstract:
        paper_kw.update(tokenize_keywords(paper.abstract))

    if not user_kw or not paper_kw:
        return 0.0

    overlap = user_kw & paper_kw
    return len(overlap) / len(user_kw)


def _category_score(paper: Paper, profile: UserProfile) -> float:
    user_cats = set(profile.interests_categories)
    overlap = user_cats & set(paper.categories)
    return len(overlap) / len(user_cats) if user_cats else 0.0


def _popularity_score(paper: Paper) -> float:
    b, v = max(paper.bookmark_count, 0), max(paper.view_count, 0)
    return (log1p(b) + log1p(v)) / 2.0


def _recency_score(paper: Paper, now: datetime) -> float:
    if not paper.update_date:
        return 0.0
    days = (now - paper.update_date).days
    return 0.5 ** (days / 730)  # 2년 반감기


def compute_total_score(paper: Paper, profile: UserProfile,
                        now: datetime = None) -> Tuple[float, Dict[str, float]]:

    now = now or datetime.utcnow()

    s_kw = _keyword_score(paper, profile)
    s_cat = _category_score(paper, profile)
    s_pop = min(_popularity_score(paper) / 10.0, 1.0)
    s_rec = _recency_score(paper, now)

    total = (WEIGHT_KEYWORD * s_kw +
             WEIGHT_CATEGORY * s_cat +
             WEIGHT_POPULARITY * s_pop +
             WEIGHT_RECENCY * s_rec)

    return total, {
        "keyword": s_kw,
        "category": s_cat,
        "popularity": s_pop,
        "recency": s_rec
    }