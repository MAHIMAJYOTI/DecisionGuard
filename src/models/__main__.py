"""Train both models and print their test metrics.

    python -m src.models
"""

from __future__ import annotations

import os

# joblib warns on this machine when it cannot see the physical core count.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from src.models.credit import train_credit_model
from src.models.fraud import train_fraud_model


def main() -> None:
    train_credit_model()
    print()
    train_fraud_model()


if __name__ == "__main__":
    main()
