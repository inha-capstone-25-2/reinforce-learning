from data_models import Paper, User, UserViewHistory, UserClickHistory, UserSearchHistory
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

def create_mock_papers() -> List[Paper]:
    """가상 논문 데이터 생성"""
    papers = []
    
    paper_templates = [
        {
            "paper_id": "0704.0001",
            "title": "Calculation of prompt diphoton production cross sections",
            "authors": ["C. Balázs", "E. L. Berger", "P. M. Nadolsky"],
            "abstract": "A fully differential calculation in perturbative quantum chromodynamics...",
            "categories": ["hep-ph", "physics"],
            "keywords": ["quantum", "photon", "colliders", "quantum chromodynamics"],
            "journal_conference": "Physical Review D",
            "difficulty_level": "advanced"
        },
        {
            "paper_id": "0704.0002",
            "title": "Deep Learning Approaches for Natural Language Processing",
            "authors": ["Y. LeCun", "Y. Bengio", "G. Hinton"],
            "abstract": "Recent advances in deep learning have revolutionized natural language processing...",
            "categories": ["cs.CL", "cs.LG", "ai"],
            "keywords": ["deep learning", "nlp", "neural networks", "transformer"],
            "journal_conference": "NeurIPS",
            "difficulty_level": "intermediate"
        },
        {
            "paper_id": "0704.0003", 
            "title": "Introduction to Machine Learning for Beginners",
            "authors": ["A. Ng", "C. Manning"],
            "abstract": "This paper provides a gentle introduction to machine learning concepts...",
            "categories": ["cs.LG", "tutorial"],
            "keywords": ["machine learning", "beginner", "tutorial", "introduction"],
            "journal_conference": "arXiv",
            "difficulty_level": "beginner"
        },
        {
            "paper_id": "0704.0004",
            "title": "Reinforcement Learning in Robotics Applications",
            "authors": ["S. Levine", "P. Abbeel"],
            "abstract": "We explore the application of reinforcement learning to robotic control...",
            "categories": ["cs.RO", "cs.LG", "ai"],
            "keywords": ["reinforcement learning", "robotics", "control", "ai"],
            "journal_conference": "ICRA",
            "difficulty_level": "advanced"
        }
    ]
    
    for template in paper_templates:
        paper = Paper(
            paper_id=template["paper_id"],
            title=template["title"],
            authors=template["authors"],
            abstract=template["abstract"],
            categories=template["categories"],
            keywords=template["keywords"],
            embedding_vector=[random.uniform(-1, 1) for _ in range(50)],  # 간단한 50차원 벡터
            publication_date="2024-01-15",
            journal_conference=template["journal_conference"],
            doi=f"10.1234/{template['paper_id']}",
            citation_count=random.randint(0, 100),
            view_count=random.randint(0, 1000),
            click_count=random.randint(0, 500),
            like_count=random.randint(0, 100),
            difficulty_level=template["difficulty_level"]
        )
        papers.append(paper)
    
    return papers

def create_mock_users() -> List[User]:
    """가상 사용자 데이터 생성"""
    users = []
    
    user_templates = [
        {
            "user_id": "user1",
            "interests": ["physics", "quantum", "hep-ph"],
            "research_field": "High Energy Physics",
            "view_history": ["0704.0001"],
            "click_history": ["0704.0001"],
            "search_queries": ["quantum chromodynamics", "photon pairs"]
        },
        {
            "user_id": "user2", 
            "interests": ["deep learning", "nlp", "ai"],
            "research_field": "Computer Science",
            "view_history": ["0704.0002"],
            "click_history": ["0704.0002"],
            "search_queries": ["transformer model", "neural networks"]
        },
        {
            "user_id": "user3",
            "interests": ["machine learning", "beginner", "tutorial"],
            "research_field": "Data Science",
            "view_history": ["0704.0003"],
            "click_history": ["0704.0003"], 
            "search_queries": ["machine learning basics", "python tutorial"]
        }
    ]
    
    for template in user_templates:
        # 뷰 히스토리 생성
        view_history = []
        for paper_id in template["view_history"]:
            view_history.append(UserViewHistory(
                paper_id=paper_id,
                timestamp="2024-01-20T10:30:00Z",
                dwell_time=random.randint(30, 300)
            ))
        
        # 클릭 히스토리 생성
        click_history = []
        for paper_id in template["click_history"]:
            click_history.append(UserClickHistory(
                paper_id=paper_id,
                timestamp="2024-01-20T10:30:00Z"
            ))
        
        # 검색 히스토리 생성
        search_history = []
        for query in template["search_queries"]:
            search_history.append(UserSearchHistory(
                query=query,
                timestamp="2024-01-20T10:30:00Z",
                result_count=random.randint(10, 50)
            ))
        
        user = User(
            user_id=template["user_id"],
            interests=template["interests"],
            research_field=template["research_field"],
            view_history=view_history,
            click_history=click_history,
            search_history=search_history,
            bookmarks=template["view_history"],  # 본 논문을 북마크했다고 가정
            preferred_categories={cat: random.uniform(0.5, 1.0) for cat in template["interests"]}
        )
        users.append(user)
    
    return users