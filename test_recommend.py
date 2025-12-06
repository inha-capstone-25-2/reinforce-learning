from recommendation.service.pipeline import recommend_for_user_hybrid

# 테스트 값
USER_ID = 2
BASE_PAPER_ID = "2401.12345"  # 상세페이지에서 보고 있다고 가정한 논문 ID
TOP_K = 6
CANDIDATE_K = 100

print("=== Hybrid Recommendation Test ===")
results = recommend_for_user_hybrid(
    user_id=USER_ID,
    top_k=TOP_K,
    candidate_k=CANDIDATE_K,
    base_paper_id=BASE_PAPER_ID,
)

for r in results:
    print(f"- {r.paper.title} | score={r.score:.4f} | similarity={r.features.get('similarity_bonus')}")