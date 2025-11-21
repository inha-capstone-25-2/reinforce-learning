from rule_based_recommender import RuleBasedRecommender
from mock_data import create_mock_papers, create_mock_users
from data_models import User

def main():
    print("룰베이스 추천 시스템 실행됨\n")
    
    # 테스트 데이터 생성
    papers = create_mock_papers()
    users = create_mock_users()
    
    # 추천 시스템 초기화
    recommender = RuleBasedRecommender()
    
    # 각 사용자별 추천 실행
    for user in users:
        print(f"\n{'='*50}")
        print(f"{user.user_id}님에게 추천하는 논문")
        print(f"관심사: {', '.join(user.interests)}")
        print(f"연구 분야: {user.research_field}")
        print(f"{'='*50}")
        
        # 추천 실행
        recommendations = recommender.recommend(user, papers, top_k=3)
        
        # 결과 출력
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}위:")
            explanation = recommender.explain_recommendation(rec)
            print(explanation)

if __name__ == "__main__":
    main()