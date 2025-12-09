from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


DEFAULT_MODEL_PATH = Path("models/rl/bandit_policy_latest.pt")


@dataclass
class PolicyConfig:
    input_dim: int
    model_path: Path = DEFAULT_MODEL_PATH
    device: str = "cpu"


class SimpleBanditModel(nn.Module if nn is not None else object):
    """
    매우 간단한 Linear 모델 (D차원 → 1차원 점수).
    - 실제 학습은 별도 trainer에서 수행하고, 여기서는 inference만 담당.
    """

    def __init__(self, input_dim: int):
        if nn is None:
            raise RuntimeError("PyTorch가 설치되어 있지 않습니다.")
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (N, D)
        return self.linear(x).squeeze(-1)  # (N,)


class BanditPolicy:
    """
    Contextual Bandit Policy

    - 학습된 PyTorch 모델(.pt)을 로딩해서
      후보 feature matrix에 대한 예상 reward score를 출력.
    - 만약 모델 파일이 없으면 -> rule-based score로 fallback.
      (state_builder에서 마지막 column을 rule_total_score로 사용하고 있으므로)
    """

    def __init__(self, config: PolicyConfig):
        self.config = config
        self.model: Optional[SimpleBanditModel] = None
        self.device = config.device
        self._loaded = False

    def load(self, input_dim: int) -> None:
        if self._loaded:
            return

        if torch is None or nn is None:
            # PyTorch 미설치 → RL 비활성화
            self.model = None
            self._loaded = True
            return

        model_path = self.config.model_path
        if not model_path.exists():
            # 모델 파일이 없으면 fallback
            self.model = None
            self._loaded = True
            return

        model = SimpleBanditModel(input_dim)
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model
        self._loaded = True

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        후보 feature matrix X(N, D)에 대해 bandit score(N,) 반환.

        - 모델이 있으면 PyTorch forward
        - 없으면 rule-based 총점 (X[:, -1])을 그대로 사용 (fallback)
        """
        if X.size == 0:
            return np.zeros((0,), dtype=float)

        # 아직 로딩 안 했으면 로딩 시도
        self.load(input_dim=X.shape[1])

        # 모델 없으면 fallback: rule-based total score 사용
        if self.model is None or torch is None:
            # 마지막 column이 rule_total_score라고 가정
            return X[:, -1].astype(float)

        with torch.no_grad():
            x_t = torch.from_numpy(X).float().to(self.device)
            scores_t = self.model(x_t)
            scores = scores_t.cpu().numpy().astype(float)
        return scores

    def select_top_k(self, X: np.ndarray, k: int) -> np.ndarray:
        """
        feature matrix X에 대해 bandit score를 계산하고 상위 k개 index 반환.
        """
        scores = self.predict_scores(X)
        if scores.size == 0:
            return np.array([], dtype=int)

        k = min(k, scores.shape[0])
        # score 내림차순 상위 k index
        top_indices = np.argsort(scores)[::-1][:k]
        return top_indices