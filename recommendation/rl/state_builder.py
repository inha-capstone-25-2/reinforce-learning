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
    """
    후보 논문들에 대해 Contextual Bandit의 입력 feature matrix를 생성.

    - 각 row: 하나의 paper
    - 각 col: 특정 feature (keyword, category, popularity, recency, rule_total_score)

    반환:
      X: (N, D) numpy array
      paper_ids: 각 row에 해당하는 paper_id 리스트
      feature_dicts: 각 paper에 대해 scoring에서 계산된 feature dict

    NOTE:
    - 여기서는 rule_based.scoring.compute_total_score 를 그대로 활용해서 feature를 뽑는다.
    - 나중에 feature를 더 추가하고 싶으면 이 함수만 수정하면 된다.
    """
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

        # 간단한 feature vector 예시 (필요시 확장 가능):
        # [keyword_score, category_score, popularity_score, recency_score, rule_total_score]
        row = [
            float(keyword),
            float(category),
            float(popularity),
            float(recency),
            float(rule_score),
        ]

        feature_rows.append(row)
        # paper_id는 arxiv_id 우선, 없으면 mongo_id 사용
        paper_ids.append(p.arxiv_id or p.mongo_id)
        feature_dicts.append({**feats, "rule_total_score": float(rule_score)})

    if not feature_rows:
        return np.zeros((0, 5), dtype=float), [], []

    X = np.asarray(feature_rows, dtype=float)
    return X, paper_ids, feature_dicts
