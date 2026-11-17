"""Compare both challengers, run the golden tests, and write the report.

    python -m src.validation
"""

from __future__ import annotations

import json
import subprocess
import sys

from src.data_gen.config import project_root
from src.validation.credit_compare import compare_credit
from src.validation.fraud_compare import compare_fraud
from src.validation.report import write_report


def main() -> None:
    root = project_root()
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    credit = compare_credit()
    (reports / "credit_compare.json").write_text(json.dumps(credit, indent=2), encoding="utf-8")
    print()
    fraud = compare_fraud()
    (reports / "fraud_compare.json").write_text(json.dumps(fraud, indent=2), encoding="utf-8")
    print()
    golden = _golden()
    markdown_path, html_path = write_report(credit, fraud, golden)
    print(f"Golden tests: {'PASS' if golden['passed'] else 'FAIL'} ({golden['summary'].strip()})")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {html_path}")


def _golden() -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_golden.py", "-q", "--tb=line"],
        cwd=project_root(),
        capture_output=True,
        text=True,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    summary = ""
    for line in reversed(output.splitlines()):
        if line.strip():
            summary = line.strip()
            break
    print(output)
    return {
        "passed": completed.returncode == 0,
        "exit_code": completed.returncode,
        "summary": summary,
        "output": output[-4000:],
    }


if __name__ == "__main__":
    main()
