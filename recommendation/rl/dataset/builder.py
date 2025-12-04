from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ...data.data_loader import MongoDataLoader
from ..state_builder import build_candidate_features
from ..utils.reward import InteractionSignal, compute_reward


@dataclass
class BanditDataset:
    X: np.ndarray
    y: np.ndarray
    user_ids: List[int]
    paper_ids: List[str]


def _iter_paper_recommendation_docs(loader: MongoDataLoader, limit=None):
    col = loader.db["paper_recommendations"]
    cursor = col.find({"recommendation_type": "rule_based"})
    if limit:
        cursor = cursor.limit(limit)

    for doc in cursor:
        yield doc


def build_bandit_dataset_from_mongo(limit: Optional[int] = None) -> BanditDataset:
    print("[DEBUG] build_bandit_dataset_from_mongo() 시작")
    t0 = time.time()

    loader = MongoDataLoader()
    print("[DEBUG] MongoDataLoader 생성 완료")

    X_rows = []
    y_list = []
    user_ids = []
    paper_ids = []

    for i, doc in enumerate(_iter_paper_recommendation_docs(loader, limit)):
        user_id = doc.get("user_id")
        paper_id = doc.get("paper_id")

        print(f"[DEBUG] doc#{i}: user={user_id}, paper={paper_id}")

        if not user_id or not paper_id:
            continue

        # --------------------------
        # PAPER 로딩 (arXiv ID 기반)
        # --------------------------
        paper = loader.get_paper_by_arxiv_id(paper_id)
        if not paper:
            print("[DEBUG]   paper 로딩 실패 → skip")
            continue

        # --------------------------
        # USER PROFILE 로딩
        # --------------------------
        profile = loader.build_user_profile(int(user_id))

        # --------------------------
        # FEATURE 생성
        # --------------------------
        X_single, pid_list, _ = build_candidate_features(profile, [paper])
        if X_single.shape[0] != 1:
            continue

        # --------------------------
        # REWARD 계산
        # --------------------------
        was_clicked = bool(doc.get("was_clicked", False))

        reward = compute_reward(
            InteractionSignal(
                was_clicked=was_clicked,
                was_bookmarked=False,
                dwell_time_ms=None
            )
        )

        X_rows.append(X_single[0])
        y_list.append(reward)
        user_ids.append(user_id)
        paper_ids.append(pid_list[0])

    print(f"[DEBUG] 샘플 개수: {len(X_rows)}")
    print(f"[DEBUG] 총 소요 시간: {time.time() - t0:.2f}초")

    if not X_rows:
        return BanditDataset(
            X=np.zeros((0, 0)),
            y=np.zeros((0,)),
            user_ids=[],
            paper_ids=[],
        )

    return BanditDataset(
        X=np.stack(X_rows),
        y=np.asarray(y_list, dtype=float),
        user_ids=user_ids,
        paper_ids=paper_ids,
    )
