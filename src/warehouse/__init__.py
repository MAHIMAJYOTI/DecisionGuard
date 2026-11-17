"""Load the raw CSV extracts into a local DuckDB warehouse.

DuckDB stands in for BigQuery. The load keeps every column as text so a bad
date or a bad number cannot crash the file landing. dbt (Phase 2) does the
typing, the rejects, and the features.
"""
