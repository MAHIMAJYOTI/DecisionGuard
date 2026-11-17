"""Generate DecisionGuard raw source files.

Run this from the project folder (the folder that contains `src/`):

    python -m src.data_gen

CSV and Parquet files are written to data/raw/, plus manifest.json which
records the row counts, the fraud rate, the booked-loan bad rate, and every
data-quality defect that was injected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_gen.config import DataGenConfig
from src.data_gen.dataset import GeneratedDataset, generate_dataset
from src.data_gen.io import default_output_dir, write_raw_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the DecisionGuard raw extracts.")
    parser.add_argument("--seed", type=int, default=42, help="Same seed rebuilds the same files.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Where to write CSV and Parquet. Default: data/raw inside the project.",
    )
    args = parser.parse_args()
    output = default_output_dir() if args.output_dir == "" else Path(args.output_dir)

    print("Building synthetic extracts. A full run is about 50,000 customers and 500,000 transactions.")
    dataset = generate_dataset(DataGenConfig(seed=args.seed))
    manifest_path = write_raw_files(dataset, output)
    _print_report(dataset, manifest_path)


def _print_report(dataset: GeneratedDataset, manifest_path: Path) -> None:
    summary = dataset.clean_summary
    print()
    print("Clean simulation (before injected defects)")
    print(f"  seed                         {summary['seed']}")
    print(f"  as-of date                   {summary['as_of']}")
    print(f"  customers                    {summary['customers']:,}")
    print(f"  applications                 {summary['applications']:,}")
    print(f"  transactions                 {summary['transactions']:,}")
    print(f"  fraud rows                   {summary['fraud_rows']:,}  ({_pct(summary['fraud_rate'])})")
    patterns = summary["fraud_patterns"]
    if isinstance(patterns, dict):
        for name, count in patterns.items():
            print(f"    {name:<28} {count:,}")
    print(f"  legitimate busy-hour txns    {summary['legit_busy_hour_txns']:,}")
    print(f"  booked loans                 {summary['booked_loans']:,}  ({_pct(summary['booked_rate'])})")
    print(f"  loans with 12 months history {summary['loans_with_12m_history']:,}")
    print(
        f"  90+ DPD in that window       {summary['dpd90_in_12m_window']:,}"
        f"  ({_pct(summary['dpd90_rate_12m_window'])})"
    )
    print(f"  early 90+ DPD (under 12m)    {summary['early_dpd90_before_12m']:,}")
    print()
    print("Defects written into the raw files")
    for issue in dataset.issues:
        print(f"  {issue.table:<20} {issue.issue:<28} {issue.rows_affected:>6,}")
    print()
    print(f"Wrote {manifest_path}")
    print("Each table is saved as both .csv and .parquet.")


def _pct(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


if __name__ == "__main__":
    main()
