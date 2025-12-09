from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..data.data_loader import MongoDataLoader
from ..models.data_models import RecommendationResult, UserProfile
from ..rl.state_builder import build_candidate_features
from ..rl.bandit_policy import SimpleBanditModel, DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # torch 미설치 환경에서는 RL 기능 비활성화


@dataclass
class RerankConfig:
    model_path: Path = Path(DEFAULT_MODEL_PATH)


class BanditPolicyWrapper:
    """
    SimpleBanditModel을 감싸는 래퍼.
    - 처음 호출 시에만 모델을 로드해서 메모리에 유지
    - 모델 파일이 없거나 torch 미설치면 None 상태로 두고, 그 경우 rule-based만 사용
    """
    def __init__(self, config: Optional[RerankConfig] = None) -> None:
        self.config = config or RerankConfig()
        self._model: Optional[SimpleBanditModel] = None

    def _ensure_model(self, input_dim: int) -> None:
        if self._model is not None:
            return

        # torch 없는 환경이면 그대로 포기
        if torch is None:
            logger.warning("[RL Reranker] ⚠️ torch 가 설치되어 있지 않아 RL rerank를 비활성화합니다.")
            return

        if not self.config.model_path.exists():
            logger.warning(f"[RL Reranker] ⚠️ RL 모델 파일이 없습니다: {self.config.model_path}")
            return

        logger.info(f"[RL Reranker] 🧠 RL 모델 로딩: {self.config.model_path} (input_dim={input_dim})")
        model = SimpleBanditModel(input_dim=input_dim)
        state = torch.load(self.config.model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        self._model = model
        logger.info("[RL Reranker] ✅ RL 모델 로딩 완료")

    def predict_scores(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        X: (N, D) feature matrix
        return: (N,) 예측 점수 or None (모델 사용 불가 시)
        """
        if X.size == 0:
            return None

        self._ensure_model(input_dim=X.shape[1])
        if self._model is None or torch is None:
            return None

        with torch.no_grad():
            t = torch.from_numpy(X).float()
            y = self._model(t).squeeze(-1).cpu().numpy()
        
        logger.info(f"[RL Reranker] 🎲 RL 점수 예측 완료: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")
        return y


class RLBanditReranker:
    """
    Rule-based 후보군(List[RecommendationResult])을 입력으로 받아
    Contextual Bandit 정책으로 rerank 후 top_k개를 반환.
    """
    def __init__(self, loader: Optional[MongoDataLoader] = None):
        self.loader = loader or MongoDataLoader()
        self.policy = BanditPolicyWrapper()

    def rerank(
        self,
        user_id: int,
        candidates: List[RecommendationResult],
        top_k: int = 6,
    ) -> List[RecommendationResult]:
        """
        candidates: RuleBasedRecommender가 만든 상위 N개 (예: 100개)
        return: RL 점수를 기준으로 다시 정렬한 상위 top_k
        """
        if not candidates:
            return []

        logger.info(f"[RL Reranker] 📥 Reranking 시작: {len(candidates)}개 후보 → top {top_k}")

        # 1) UserProfile 로딩
        profile: UserProfile = self.loader.build_user_profile(user_id)

        # 2) 후보 논문들 feature matrix 생성
        papers = [c.paper for c in candidates]
        X, paper_ids, _feat_dicts = build_candidate_features(profile, papers)
        logger.info(f"[RL Reranker] 📊 Feature matrix 생성: shape={X.shape}")

        # 3) RL 정책으로 점수 예측
        rl_scores = self.policy.predict_scores(X)

        # RL 사용 불가(troch 미설치, 모델 없음 등) → rule-based 순서 그대로 top_k
        if rl_scores is None:
            logger.warning("[RL Reranker] ⚠️ RL 모델 사용 불가 → rule-based 결과 그대로 사용")
            return candidates[:top_k]

        logger.info("[RL Reranker] ✅ RL 모델 활성화 → RL 점수로 reranking")

        # 4) 후보들에 RL 점수 적용
        reranked: List[RecommendationResult] = []
        for c, rl_s in zip(candidates, rl_scores):
            # 기존 rule-based score를 보존하고, score를 RL 점수로 덮어씌움
            rule_score = c.score
            c.features = dict(c.features or {})
            c.features["rule_score"] = float(rule_score)
            c.features["rl_score"] = float(rl_s)
            c.score = float(rl_s)
            reranked.append(c)

        # 5) RL 점수 기준 내림차순 정렬 후 top_k
        reranked.sort(key=lambda r: r.score, reverse=True)
        
        logger.info(f"[RL Reranker] 📤 Reranking 완료: {len(reranked[:top_k])}개 반환")
        return reranked[:top_k]