from typing import List, Dict
from data_models import Paper, User

class RuleBasedScorer:
    @staticmethod
    def calculate_interest_score(user: User, paper: Paper) -> float:
        """관심사 기반 점수 계산"""
        score = 0.0
        
        # 키워드 매칭
        for user_interest in user.interests:
            # 키워드 직접 매칭
            if user_interest in paper.keywords:
                score += 3.0
            
            # 카테고리 매칭
            if user_interest in paper.categories:
                score += 2.0
            
            # 제목에 키워드 포함
            if user_interest.lower() in paper.title.lower():
                score += 1.0
        
        return score

    @staticmethod
    def calculate_popularity_score(paper: Paper) -> float:
        """인기도 기반 점수 계산"""
        score = 0.0
        
        # 조회수 가중치
        score += paper.view_count * 0.001
        # 클릭수 가중치
        score += paper.click_count * 0.002
        # 인용수 가중치
        score += paper.citation_count * 0.01
        # 좋아요 수
        score += paper.like_count * 0.005
        
        return score

    @staticmethod
    def calculate_recency_score(paper: Paper) -> float:
        """최신 점수 계산"""
        # 일단은 이렇게만 (실제로는 publication_date 파싱 필요)
        return 1.0  # 모든 논문에 동일하게 부여

    @staticmethod
    def calculate_personalization_score(user: User, paper: Paper) -> float:
        """개인화 점수 계산"""
        score = 0.0
        
        # 연구 분야 매칭
        if user.research_field.lower() in paper.abstract.lower():
            score += 2.0
        
        # 선호 카테고리 가중치
        for category in paper.categories:
            if category in user.preferred_categories:
                score += user.preferred_categories[category]
        
        # 북마크/히스토리 기반
        viewed_papers = [vh.paper_id for vh in user.view_history]
        if paper.paper_id in viewed_papers:
            score -= 5.0  # 이미 본 논문은 점수 감점
        
        return score