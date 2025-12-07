from .interface.recommend import (
    recommend_user,
    recommend_user_rl,
)

def demo_user_recommendation(user_id: int = 1):
    print(f"=== User {user_id} Recommendations (Rule-based) ===")
    recs = recommend_user(user_id, top_k=6)
    for r in recs:
        print(f"{r['title']} (score={r['score']:.4f})")

def demo_user_recommendation_rl(user_id: int = 1):
    print(f"=== User {user_id} Recommendations (Rule+RL) ===")
    recs = recommend_user_rl(user_id, top_k=6, candidate_k=100)
    for r in recs:
        print(f"{r['title']} (score={r['score']:.4f})")

if __name__ == "__main__":
    demo_user_recommendation(user_id=1)
    demo_user_recommendation_rl(user_id=1)