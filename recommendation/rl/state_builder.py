# recommendation/rl/state_builder.py

from typing import List, Tuple, Dict
import numpy as np
from datetime import datetime

from ..models.data_models import Paper, UserProfile
from ..rule_based.scoring import compute_total_score


def build_candidate_features(
    profile: UserProfile,
    candidates: List[Paper],
    now: datetime | None = None,
) -> Tuple[np.ndarray, List[str], List[Dict[str, float]]]:
    
    if now is None:
        now = datetime.utcnow()

    feature_rows: List[List[float]] = []
    paper_ids: List[str] = []
    feature_dicts: List[Dict[str, float]] = []

    for p in candidates:
        rule_score, feats = compute_total_score(p, profile, now=now)

        # feats: {"keyword": s_kw, "category": s_cat, "popularity": s_pop, "recency": s_rec}
        # feature vector 구성 순서를 고정해두자.
        keyword = feats.get("keyword", 0.0)
        category = feats.get("category", 0.0)
        popularity = feats.get("popularity", 0.0)
        recency = feats.get("recency", 0.0)

        row = [
            float(keyword),
            float(category),
            float(popularity),
            float(recency),
            float(rule_score),
        ]

        feature_rows.append(row)
        paper_ids.append(p.arxiv_id or p.mongo_id)
        feature_dicts.append({**feats, "rule_total_score": float(rule_score)})

    if not feature_rows:
        return np.zeros((0, 5), dtype=float), [], []

    X = np.asarray(feature_rows, dtype=float)
    return X, paper_ids, feature_dicts
