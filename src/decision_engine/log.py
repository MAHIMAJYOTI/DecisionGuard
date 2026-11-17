"""Append-only audit log. Rows are added, never updated.

The input snapshot is the JSON of the row the strategy saw, so a later
change to the mart does not rewrite history.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def write_decisions(
    db_path: Path,
    source: pd.DataFrame,
    decided: pd.DataFrame,
    run_id: str,
) -> int:
    rows = source.reset_index(drop=True)
    if "entity_id" in decided.columns:
        entity_ids = decided["entity_id"].astype(str)
    elif "application_id" in rows.columns:
        entity_ids = rows["application_id"].astype(str)
    elif "txn_id" in rows.columns:
        entity_ids = rows["txn_id"].astype(str)
    else:
        entity_ids = [f"row-{index}" for index in range(len(decided))]
    snapshots = [
        json.dumps(_json_ready(record), sort_keys=True, default=str)
        for record in rows.to_dict(orient="records")
    ]
    log_frame = pd.DataFrame(
        {
            "decision_id": [f"{run_id}-{index}" for index in range(len(decided))],
            "run_id": run_id,
            "entity_type": np.where(
                decided["strategy_version"].astype(str).str.startswith("fraud").to_numpy(),
                "transaction",
                "application",
            ),
            "entity_id": entity_ids,
            "decision": decided["decision"].astype(str),
            "reason_codes": [json.dumps(list(reasons)) for reasons in decided["reason_codes"]],
            "score": pd.to_numeric(decided["score"], errors="coerce"),
            "strategy_version": decided["strategy_version"].astype(str),
            "model_version": decided["model_version"].astype(str),
            "credit_limit": pd.to_numeric(decided["credit_limit"], errors="coerce"),
            "input_snapshot": snapshots,
            "decided_at": datetime.now(timezone.utc),
        }
    )
    connection = duckdb.connect(str(db_path))
    try:
        connection.execute("CREATE SCHEMA IF NOT EXISTS decisions")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions.decision_log (
                decision_id VARCHAR,
                run_id VARCHAR,
                entity_type VARCHAR,
                entity_id VARCHAR,
                decision VARCHAR,
                reason_codes VARCHAR,
                score DOUBLE,
                strategy_version VARCHAR,
                model_version VARCHAR,
                credit_limit DOUBLE,
                input_snapshot VARCHAR,
                decided_at TIMESTAMP
            )
            """
        )
        connection.register("incoming_decisions", log_frame)
        connection.execute("INSERT INTO decisions.decision_log SELECT * FROM incoming_decisions")
    finally:
        connection.close()
    return len(log_frame)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return float(value)
    if value is None or (isinstance(value, str) and value == ""):
        return value
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value
