from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


"""
Offline 학습용 trainer 모듈 패키지
"""


try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise ImportError(
        "PyTorch가 설치되어 있어야 offline RL 학습을 수행할 수 있습니다. "
        "pip install torch 로 설치 후 다시 시도하세요."
    ) from e

from ..bandit_policy import SimpleBanditModel, DEFAULT_MODEL_PATH
from ..dataset.builder import build_bandit_dataset_from_mongo


def train_offline_bandit(
    model_path: Optional[Path] = None,
    batch_size: int = 256,
    num_epochs: int = 5,
    lr: float = 1e-3,
    limit: Optional[int] = None,
) -> Path:
    """
    MongoDB paper_recommendations 데이터를 사용하여
    SimpleBanditModel을 offline 학습하고, model_path에 저장한다.
    """
    model_path = Path(model_path or DEFAULT_MODEL_PATH)

    # 1) Dataset 구축
    dataset = build_bandit_dataset_from_mongo(limit=limit)
    X, y = dataset.X, dataset.y

    if X.size == 0 or y.size == 0:
        raise RuntimeError(
            "offline 학습용 데이터가 없습니다. "
            "paper_recommendations 컬렉션에 데이터가 쌓였는지 확인하세요."
        )

    input_dim = X.shape[1]

    # 2) TensorDataset 준비
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()

    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # 3) 모델 초기화
    model = SimpleBanditModel(input_dim=input_dim)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 회귀 문제로 간주하여 MSE 사용 (reward ~ expected value)
    criterion = nn.MSELoss()

    # 4) 학습 루프
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for xb, yb in dl:
            optimizer.zero_grad()
            pred = model(xb)  # (B,)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * xb.size(0)

        epoch_loss /= len(ds)
        print(f"[epoch {epoch+1}/{num_epochs}] loss={epoch_loss:.4f}")

    # 5) 디렉토리 생성 및 저장
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved bandit policy model to: {model_path}")

    return model_path


if __name__ == "__main__":
    # 예시 실행:
    train_offline_bandit()