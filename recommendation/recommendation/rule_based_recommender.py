from typing import List, Dict, Tuple
from data_models import Paper, User
from utils.scoring import RuleBasedScorer

class RuleBasedRecommender:
    def __init__(self):
        self.scorer = RuleBasedScorer()
    
    def recommend(self, user: User, papers: List[Paper], top_k: int = 10) -> List[Dict]:
        """룰베이스 추천 실행"""
        recommendations = []
        
        for paper in papers:
            # 각 점수 계산 
            interest_score = self.scorer.calculate_interest_score(user, paper)
            popularity_score = self.scorer.calculate_popularity_score(paper)
            recency_score = self.scorer.calculate_recency_score(paper)
            personalization_score = self.scorer.calculate_personalization_score(user, paper)
            
            # 최종 점수 (가중치 조정 가능)
            total_score = (
                interest_score * 0.4 +
                popularity_score * 0.2 +
                recency_score * 0.1 +
                personalization_score * 0.3
            )
            
            # 추천 이유 분석
            reasons = self._analyze_recommendation_reasons(
                interest_score, popularity_score, personalization_score
            )
            
            recommendations.append({
                "paper": paper,
                "total_score": total_score,
                "breakdown": {
                    "interest_score": interest_score,
                    "popularity_score": popularity_score,
                    "recency_score": recency_score,
                    "personalization_score": personalization_score
                },
                "reasons": reasons,
                "paper_id": paper.paper_id
            })
        
        # 점수 기준 정렬 및 상위 k개 선택
        recommendations.sort(key=lambda x: x["total_score"], reverse=True)
        return recommendations[:top_k]
    
    def _analyze_recommendation_reasons(self, interest_score: float, 
                                      popularity_score: float, 
                                      personalization_score: float) -> List[str]:
        """추천 이유 분석"""
        reasons = []
        
        if interest_score > 2.0:
            reasons.append("관심사와 높은 관련성")
        elif interest_score > 0.5:
            reasons.append("관심사와 일부 관련성")
            
        if popularity_score > 1.0:
            reasons.append("인기 논문")
            
        if personalization_score > 1.0:
            reasons.append("개인 취향과 일치")
            
        return reasons if reasons else ["다양한 주제의 논문"]
    
    def explain_recommendation(self, recommendation: Dict) -> str:
        """추천 결과 설명 생성"""
        paper = recommendation["paper"]
        reasons = recommendation["reasons"]
        
        explanation = f"📄 '{paper.title}'\n"
        explanation += f"총점: {recommendation['total_score']:.2f}\n"
        explanation += f"추천 이유: {', '.join(reasons)}\n"
        explanation += f"분야: {', '.join(paper.categories)}\n"
        explanation += f"인기도: 조회수 {paper.view_count}, 인용 {paper.citation_count}\n"
        
        return explanation