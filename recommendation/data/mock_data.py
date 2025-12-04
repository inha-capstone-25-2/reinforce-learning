from datetime import datetime
from typing import List

from ..models.data_models import Paper

def get_mock_papers() -> List[Paper]:
    return [
        Paper(
            mongo_id="mock-1",
            arxiv_id="0001.0001",
            title="Mock Paper 1",
            abstract="Mock abstract about RL.",
            authors="John Doe",
            categories=["cs.LG"],
            keywords=["ppo", "rl"],
            update_date=datetime(2024, 1, 1),
            bookmark_count=10,
            view_count=100,
            difficulty_level="intermediate",
            summary={"en": "Short English summary."}
        ),
    ]