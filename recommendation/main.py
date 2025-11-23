from .interface.recommend import recommend_user, recommend_similar_papers


def demo_user_recommendation(user_id: int = 1):
    print(f"=== User {user_id} Recommendations ===")
    recs = recommend_user(user_id, top_k=6)
    for r in recs:
        print(f"{r['title']} (score={r['score']:.4f})")


def demo_similar_paper(paper_id: str = "0704.0001"):
    print(f"=== Similar Papers for {paper_id} ===")
    recs = recommend_similar_papers(paper_id, top_k=6)
    for r in recs:
        print(f"{r['title']} (score={r['score']:.4f})")


if __name__ == "__main__":
    demo_user_recommendation(user_id=1)
    demo_similar_paper("0704.0001")