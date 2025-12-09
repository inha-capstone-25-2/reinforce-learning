from recommendation.service.pipeline import recommend_for_user_hybrid


USER_ID = 2  # 테스트할 유저 ID 

# 1) 자기 자신이 추천에 포함되지 않는지 확인
def test_self_exclusion(base_paper_id: str):
    print(f"\n=== [Self Exclusion Test] base_paper_id = {base_paper_id} ===")
    recs = recommend_for_user_hybrid(
        user_id=USER_ID,
        top_k=6,
        candidate_k=100,
        base_paper_id=base_paper_id,
    )

    if not recs:
        print("→ 추천 결과가 없습니다.")
        return

    has_self = False
    for idx, r in enumerate(recs, start=1):
        pid = getattr(r.paper, "arxiv_id", None)
        cats = getattr(r.paper, "categories", [])
        sim = r.features.get("similarity_bonus")
        print(f"{idx}. {pid} | title={r.paper.title[:60]}... | sim={sim} | cats={cats}")
        if pid == base_paper_id:
            has_self = True

    if has_self:
        print("⚠️  자기 자신 논문이 추천에 포함되어 있음 → 필터링 로직 확인 필요")
    else:
        print("✅  자기 자신 논문은 추천 결과에 포함되지 않음 (정상)")


# 2) 여러 다른 논문에 대해 추천 구성이 얼마나 달라지는지 체크
def test_diversity(base_paper_ids):
    for base_paper_id in base_paper_ids:
        print(f"\n=== [Diversity Test] base_paper_id = {base_paper_id} ===")
        recs = recommend_for_user_hybrid(
            user_id=USER_ID,
            top_k=6,
            candidate_k=100,
            base_paper_id=base_paper_id,
        )

        if not recs:
            print("→ 추천 결과가 없습니다.")
            continue

        seen_ids = set()
        cat_counts = {}

        for idx, r in enumerate(recs, start=1):
            pid = getattr(r.paper, "arxiv_id", None)
            cats = getattr(r.paper, "categories", []) or []
            sim = r.features.get("similarity_bonus")
            rl_score = r.features.get("rl_score")
            print(f"{idx}. {pid} | rl={rl_score:.4f} | sim={sim} | cats={cats}")

            seen_ids.add(pid)
            for c in cats:
                cat_counts[c] = cat_counts.get(c, 0) + 1

        print(f" → 추천된 논문 개수: {len(recs)}, 서로 다른 논문 ID 수: {len(seen_ids)}")
        print(f" → 카테고리 분포: {cat_counts}")


if __name__ == "__main__":
    # 🔹 1) 자기 자신이 추천에 뜨는지 테스트할 논문 ID 하나
    base_paper_for_self_test = ["2401.12345", "2308.77777", "2410.99999"]   # 실제 존재하는 arxiv_id로 교체

    # 🔹 2) 서로 다른 상세페이지 몇 개에 대해 diversity 테스트
    base_papers_for_div = [
        "2401.12345",   # 예: RL 논문
        "2403.56789",   # 예: GNN 논문
        "2312.00001",   # 예: CV 논문
    ]  # 네 DB에 실제로 있는 ID들로 교체

    test_self_exclusion(base_paper_for_self_test)
    test_diversity(base_papers_for_div)