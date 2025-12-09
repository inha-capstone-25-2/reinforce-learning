"""
RL 추천 서버 - FastAPI 메인 파일.

Rule-based + RL(Contextual Bandit) 기반 논문 추천 API를 제공합니다.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from recommendation.interface.api_interface import (
    get_user_recommendations,
    get_similar_paper_recommendations,
    get_user_recommendations_rl,
    log_recommendation_interaction,
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --- Schemas (백엔드 형식에 맞춤) ---


class PaperRecommendation(BaseModel):
    """개별 추천 논문"""
    paper_id: str
    title: Optional[str] = None
    authors: Optional[str] = None
    abstract: Optional[str] = None
    categories: List[str] = []
    summary: Optional[Dict[str, Any]] = None
    external_url: Optional[str] = None
    total_score: float
    breakdown: Dict[str, float] = {}


class RecommendationResponse(BaseModel):
    """추천 API 응답 (백엔드 규격)"""
    user_id: int
    session_id: str
    recommendation_type: str
    recommendations: List[PaperRecommendation]
    total_count: int
    timestamp: str


class SimilarPaperResponse(BaseModel):
    """유사 논문 추천 응답"""
    paper_id: str
    session_id: str
    recommendation_type: str
    recommendations: List[PaperRecommendation]
    total_count: int
    timestamp: str


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    service: str
    version: str


class InteractionRequest(BaseModel):
    """상호작용 로그 요청"""
    user_id: int
    paper_id: str
    action_type: str = "click"  # "click" | "bookmark"
    recommendation_id: Optional[str] = None
    position: Optional[int] = None
    dwell_time: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


class InteractionResponse(BaseModel):
    """상호작용 로그 응답"""
    ok: bool
    interaction_id: str
    reward: float


# --- Helper Functions ---


def transform_paper(raw_paper: Dict[str, Any]) -> PaperRecommendation:
    """기존 응답 형식을 백엔드 규격으로 변환"""
    return PaperRecommendation(
        paper_id=raw_paper.get("id", ""),
        title=raw_paper.get("title"),
        authors=raw_paper.get("authors"),
        abstract=raw_paper.get("abstract"),
        categories=raw_paper.get("categories", []),
        summary=raw_paper.get("summary"),
        external_url=raw_paper.get("externalUrl"),
        total_score=raw_paper.get("score", 0.0),
        breakdown=raw_paper.get("features", {}),
    )


def create_session_id() -> str:
    """새 세션 ID 생성"""
    return str(uuid.uuid4())


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 및 종료"""
    logger.info("[Startup] RL Recommendation Server starting...")
    
    # 추천 시스템 초기화 (lazy loading이지만 시작 시 워밍업)
    try:
        # 간단한 워밍업 호출로 MongoDB 연결 확인
        logger.info("[Startup] Warming up recommendation system...")
        get_user_recommendations(user_id=1, limit=1)
        logger.info("[Startup] Recommendation system ready")
    except Exception as e:
        logger.warning(f"[Startup] Warmup failed (will retry on first request): {e}")
    
    yield
    
    logger.info("[Shutdown] RL Recommendation Server shutting down...")


app = FastAPI(
    title="RL Recommendation Server",
    description="Rule-based + RL(Contextual Bandit) 기반 논문 추천 API",
    version="1.0.0",
    lifespan=lifespan,
)


# --- API Endpoints ---


@app.get("/")
def root():
    """루트 엔드포인트"""
    return {
        "message": "RL Recommendation Server",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/recommendations",
            "/recommendations/rl",
            "/recommendations/similar/{paper_id}",
            "/recommendations/interactions",
        ],
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크"""
    return HealthResponse(
        status="ok",
        service="rl-recommendation",
        version="1.0.0",
    )


@app.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: int = Query(..., description="사용자 ID"),
    limit: int = Query(6, ge=1, le=50, description="추천 개수"),
    session_id: Optional[str] = Query(None, description="세션 ID (없으면 자동 생성)"),
):
    """
    Rule-based 추천.
    
    사용자의 관심사, 북마크 기록, 검색 기록 기반으로 추천합니다.
    """
    try:
        logger.info(f"[API] Rule-based recommendations: user_id={user_id}, limit={limit}")
        
        # 기존 api_interface 호출
        raw_result = get_user_recommendations(user_id=user_id, limit=limit)
        
        # 응답 변환
        session = session_id or create_session_id()
        recommendations = [transform_paper(p) for p in raw_result.get("results", [])]
        
        response = RecommendationResponse(
            user_id=user_id,
            session_id=session,
            recommendation_type="rule_based",
            recommendations=recommendations,
            total_count=len(recommendations),
            timestamp=datetime.utcnow().isoformat(),
        )
        
        logger.info(f"[API] Returned {len(recommendations)} recommendations")
        return response
        
    except Exception as e:
        logger.error(f"[API] Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"추천 생성 실패: {str(e)}")


@app.get("/recommendations/rl", response_model=RecommendationResponse)
async def get_recommendations_rl(
    user_id: int = Query(..., description="사용자 ID"),
    limit: int = Query(6, ge=1, le=50, description="최종 추천 개수"),
    candidate_k: int = Query(100, ge=10, le=500, description="RL reranking 후보군 크기"),
    base_paper_id: Optional[str] = Query(None, description="현재 보고 있는 논문 ID (유사도 보너스 계산용)"),
    session_id: Optional[str] = Query(None, description="세션 ID (없으면 자동 생성)"),
):
    """
    RL (Contextual Bandit) 기반 추천.
    
    1. Rule-based로 candidate_k개 후보 생성
    2. base_paper_id가 있으면 유사도 보너스 적용
    3. RL 모델로 reranking
    4. 최종 limit개 반환
    """
    try:
        logger.info(f"[API] RL recommendations: user_id={user_id}, limit={limit}, candidate_k={candidate_k}, base_paper_id={base_paper_id}")
        
        # 기존 api_interface 호출
        raw_result = get_user_recommendations_rl(
            user_id=user_id,
            limit=limit,
            candidate_k=candidate_k,
            base_paper_id=base_paper_id,
        )
        
        # 응답 변환
        session = session_id or create_session_id()
        recommendations = [transform_paper(p) for p in raw_result.get("results", [])]
        
        response = RecommendationResponse(
            user_id=user_id,
            session_id=session,
            recommendation_type="rl_based",
            recommendations=recommendations,
            total_count=len(recommendations),
            timestamp=datetime.utcnow().isoformat(),
        )
        
        logger.info(f"[API] Returned {len(recommendations)} RL recommendations")
        return response
        
    except Exception as e:
        logger.error(f"[API] RL Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"RL 추천 생성 실패: {str(e)}")


@app.get("/recommendations/similar/{paper_id}", response_model=SimilarPaperResponse)
async def get_similar_papers(
    paper_id: str,
    limit: int = Query(6, ge=1, le=50, description="추천 개수"),
    session_id: Optional[str] = Query(None, description="세션 ID (없으면 자동 생성)"),
):
    """
    유사 논문 추천.
    
    특정 논문과 유사한 논문을 추천합니다.
    """
    try:
        logger.info(f"[API] Similar papers: paper_id={paper_id}, limit={limit}")
        
        # 기존 api_interface 호출
        raw_result = get_similar_paper_recommendations(paper_id=paper_id, limit=limit)
        
        # 응답 변환
        session = session_id or create_session_id()
        recommendations = [transform_paper(p) for p in raw_result.get("results", [])]
        
        response = SimilarPaperResponse(
            paper_id=paper_id,
            session_id=session,
            recommendation_type="similar_papers",
            recommendations=recommendations,
            total_count=len(recommendations),
            timestamp=datetime.utcnow().isoformat(),
        )
        
        logger.info(f"[API] Returned {len(recommendations)} similar papers")
        return response
        
    except Exception as e:
        logger.error(f"[API] Similar papers error: {e}")
        raise HTTPException(status_code=500, detail=f"유사 논문 추천 실패: {str(e)}")


@app.post("/recommendations/interactions", response_model=InteractionResponse)
async def log_interaction(request: InteractionRequest):
    """
    추천 상호작용 로그 기록.
    
    사용자가 추천된 논문을 클릭하거나 북마크할 때 호출.
    reward 값을 계산하여 반환합니다.
    
    - click: +1.0
    - bookmark: +3.0
    - dwell_time >= 3초: +0.3
    - dwell_time <= 1초: -0.2 (이탈 페널티)
    """
    try:
        logger.info(f"[API] Interaction log: user_id={request.user_id}, paper_id={request.paper_id}, action={request.action_type}")
        
        result = log_recommendation_interaction(
            user_id=request.user_id,
            paper_id=request.paper_id,
            action_type=request.action_type,
            recommendation_id=request.recommendation_id,
            position=request.position,
            dwell_time=request.dwell_time,
            meta=request.meta,
        )
        
        logger.info(f"[API] Interaction logged: interaction_id={result.get('interaction_id')}, reward={result.get('reward')}")
        
        return InteractionResponse(
            ok=result.get("ok", True),
            interaction_id=result.get("interaction_id", ""),
            reward=result.get("reward", 0.0),
        )
        
    except Exception as e:
        logger.error(f"[API] Interaction log error: {e}")
        raise HTTPException(status_code=500, detail=f"상호작용 로그 실패: {str(e)}")

