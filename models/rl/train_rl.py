import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os
import json

# --------------------------
#  MongoDB 설정
# --------------------------
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "recommendation_db"

# --------------------------
# Bandit Policy 모델 정의
# --------------------------
class BanditPolicy(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),     # score output
        )

    def forward(self, x):
        return self.net(x)


# --------------------------
# MongoDB에서 로그 가져오기 (JOIN 포함)
# --------------------------
def load_logs_from_mongo():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    events = list(db.recommendation_events.find({}))
    interactions = list(db.recommendation_interactions.find({}))

    event_map = {e["recommendation_id"]: e for e in events}

    samples = []

    for inter in interactions:
        rec_id = inter["recommendation_id"]
        if rec_id not in event_map:
            continue

        clicked_paper = inter["paper_id"]
        reward = inter["reward"]

        event = event_map[rec_id]

        # 추천된 목록 중 클릭된 논문 찾기
        for item in event["results"]:
            if item["id"] == clicked_paper:
                features = item["features"]  # rule_score 포함
                samples.append((features, reward))
                break

    print(f"Loaded logs: {len(samples)} samples")
    return samples


# --------------------------
# feature → tensor 변환
# --------------------------
def convert_to_tensor(samples):
    X = []
    y = []

    for feat, reward in samples:
        vector = [
            feat.get("recency", 0),
            feat.get("popularity", 0),
            feat.get("category", 0),
            feat.get("keyword", 0),
            feat.get("rule_score", 0),
        ]
        X.append(vector)
        y.append(reward)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    return X, y


# --------------------------
# 모델 학습
# --------------------------
def train_model(X, y, epochs=100, lr=1e-3):
    model = BanditPolicy(input_dim=5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Loss = {loss.item():.4f}")

    return model


# --------------------------
# 모델 저장
# --------------------------
def save_model(model):
    os.makedirs("models/rl", exist_ok=True)
    save_path = "models/rl/bandit_policy_latest.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at: {save_path}")


# --------------------------
# 실행 메인 함수
# --------------------------
if __name__ == "__main__":
    print("🚀 RL Training Start")

    samples = load_logs_from_mongo()
    if len(samples) == 0:
        print("No training samples available. Stop.")
        exit()

    X, y = convert_to_tensor(samples)
    model = train_model(X, y, epochs=80)
    save_model(model)

    print("🎉 RL Training Finished Successfully!")
