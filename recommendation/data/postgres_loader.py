from __future__ import annotations

import os
from typing import List

import psycopg2
from psycopg2.extras import RealDictCursor


PG_HOST = os.getenv("PG_HOST", "35.94.93.225")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DBNAME = os.getenv("PG_DBNAME", "rsrs")
PG_USER = os.getenv("PG_USER", "rsrs-root")
PG_PASSWORD = os.getenv("PG_PASSWORD", "e2XNR0qnZ7kKygC3Sl5zQ2BF2FkHcCr110CaCqulOOlPs")


class PostgresUserInterestLoader:
    """
    PostgreSQL에서 사용자 관심 카테고리(user_interests + categories.code)를 로드하는 헬퍼.

    - user_interests(user_id, category_id)
    - categories(id, code, ...)
    """

    def __init__(self):
        self._conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DBNAME,
            user=PG_USER,
            password=PG_PASSWORD,
            cursor_factory=RealDictCursor,
        )

    def get_user_category_codes(self, user_id: int) -> List[str]:
        """
        주어진 user_id에 대해 categories.code 리스트를 반환.

        예: ["cs.LG", "stat.ML", "physics", ...]
        """
        query = """
        SELECT c.code
        FROM user_interests ui
        JOIN categories c ON ui.category_id = c.id
        WHERE ui.user_id = %s
        """
        with self._conn.cursor() as cur:
            cur.execute(query, (user_id,))
            rows = cur.fetchall()

        codes = []
        for row in rows:
            code = row.get("code")
            if code:
                codes.append(code)
        # 중복 제거
        return sorted(set(codes))

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass