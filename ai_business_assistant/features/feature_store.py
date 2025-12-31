"""
Feature store for computed features with versioning and DuckDB backend.
"""

import duckdb
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional
from ai_business_assistant.config import get_settings

settings = get_settings()

class FeatureStore:
    def __init__(self, db_path: str = str(settings.abs_warehouse_path)):
        self.conn = duckdb.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_metadata (
                feature_name VARCHAR,
                version VARCHAR,
                computed_at TIMESTAMP,
                lineage_info JSON,
                PRIMARY KEY(feature_name, version)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_values (
                feature_name VARCHAR,
                entity_id VARCHAR,
                value DOUBLE,
                version VARCHAR,
                timestamp TIMESTAMP
            )
        """)

    def store_feature(self, name: str, version: str, data: pd.DataFrame, lineage: Optional[Dict] = None):
        """
        Store computed features.
        data should have columns: entity_id, value, timestamp
        """
        # Store metadata
        import json
        self.conn.execute(
            "INSERT OR REPLACE INTO feature_metadata VALUES (?, ?, ?, ?)",
            (name, version, datetime.now(), json.dumps(lineage) if lineage else "{}")
        )
        
        # Store values
        # For simplicity, we'll append to the table
        self.conn.execute("INSERT INTO feature_values SELECT ?, entity_id, value, ?, timestamp FROM data", (name, version))

    def get_latest_feature_values(self, name: str) -> pd.DataFrame:
        return self.conn.execute(
            "SELECT * FROM feature_values WHERE feature_name = ? QUALIFY row_number() OVER (PARTITION BY entity_id ORDER BY timestamp DESC) = 1",
            (name,)
        ).df()

    def get_feature_history(self, name: str, entity_id: str) -> pd.DataFrame:
        return self.conn.execute(
            "SELECT * FROM feature_values WHERE feature_name = ? AND entity_id = ? ORDER BY timestamp DESC",
            (name, entity_id)
        ).df()

    def get_feature_stats(self, name: str) -> Dict[str, Any]:
        result = self.conn.execute(
            "SELECT COUNT(*), AVG(value), MIN(value), MAX(value), STDDEV(value) FROM feature_values WHERE feature_name = ?",
            (name,)
        ).fetchone()
        return {
            "count": result[0],
            "avg": result[1],
            "min": result[2],
            "max": result[3],
            "stddev": result[4]
        }
