"""Knobs for the synthetic portfolio.

Change a number here if you want a smaller practice run. Leave the defaults
when you want the portfolio described in the project brief: 50,000 customers,
50,000 applications, about 500,000 card transactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# These two sizes are what the defect counts in quality.py were designed for.
# A smaller run scales those defects down instead of copying all 750 nulls
# into a tiny file.
DEFAULT_N_CUSTOMERS = 50_000
DEFAULT_N_APPLICATIONS = 50_000
DEFAULT_N_TRANSACTIONS = 500_000

# One frozen "today" so a rerun next month still builds the same dates.
AS_OF = "2026-10-01"

# Applications are spread over the two years before that date, so some loans
# have a full 12 months of history and some are still too new to judge.
APPLICATION_START = "2024-10-01"
APPLICATION_DAYS = 730

# Ninety days of card spend is enough for "last hour", "last day" and
# "last 30 days" features, and it still fits on a laptop.
TRANSACTION_START = "2026-07-03"
TRANSACTION_DAYS = 90

FRAUD_RATE = 0.005

# Share of each fraud *row* (they sum to 1). The generator turns these into
# exact row counts. See transactions.py for what each pattern looks like.
FRAUD_SHARES: dict[str, float] = {
    "fraud_burst": 0.28,
    "fraud_foreign_new_device": 0.24,
    "fraud_night_high_value": 0.18,
    "fraud_amount_spike": 0.15,
    "fraud_stealth": 0.15,
}

# About 45% of customers have a card that actually spends. The rest are
# savings / loan-only customers and do not appear in the transaction file.
CARDHOLDER_FRACTION = 0.45

PRODUCTS = ("personal loan", "credit card")
CHANNELS = ("branch", "web", "app")
EMPLOYMENT_TYPES = (
    "salaried",
    "self_employed",
    "student",
    "retired",
    "unemployed",
    "homemaker",
)

# ISO country codes, the way a card processor sends them.
HOME_COUNTRY = "IN"
TRAVEL_COUNTRIES = ("AE", "TH", "SG", "GB", "US")
FRAUD_COUNTRIES = ("NG", "US", "GB", "AE", "RO", "BR")

# Values that are not real dates. Staging (Phase 2) must reject them.
BAD_DATE_TOKENS = (
    "2024-13-45",
    "not-a-date",
    "1900-01-01 00:00:00",
    "2099-01-01 00:00:00",
)

# CUST0000000 is never issued. Applications that point at it are broken keys.
ORPHAN_CUSTOMER_ID = "CUST0000000"
ORPHAN_APPLICATION_ID = "APP00000000"

CUSTOMER_COLUMNS = (
    "customer_id",
    "age",
    "income",
    "employment_type",
    "city",
    "existing_debt",
    "months_at_address",
)

BUREAU_COLUMNS = (
    "customer_id",
    "bureau_score",
    "num_enquiries_6m",
    "num_delinquencies_24m",
    "total_credit_limit",
    "utilisation",
)

APPLICATION_COLUMNS = (
    "application_id",
    "customer_id",
    "product",
    "requested_amount",
    "tenure_months",
    "application_ts",
    "channel",
)

TRANSACTION_COLUMNS = (
    "txn_id",
    "customer_id",
    "amount",
    "merchant_category",
    "country",
    "device_id",
    "txn_ts",
    "is_fraud",
)

# A servicing extract, not an application form. Only booked loans appear.
# dpd90_within_12m is blank when 12 months have not passed and the loan has
# not already reached 90 days past due. That blank means "not known yet".
PERFORMANCE_COLUMNS = (
    "application_id",
    "customer_id",
    "disbursed_amount",
    "disbursement_ts",
    "months_on_book",
    "max_dpd_12m",
    "dpd90_within_12m",
)

# Historical booking rules. These are NOT the credit strategy in strategies/.
# They only decide which synthetic loans have an outcome to learn from.
# A real bank has the same gap: you cannot observe default on a loan you declined.
BOOK_MIN_AGE = 21
BOOK_MIN_SCORE = 580
BOOK_MAX_DTI = 0.60
BOOK_MAX_DELINQUENCIES = 3  # 3 or more is a decline
BOOK_MIN_INCOME = 180_000  # annual INR
BOOK_REFER_SCORE = 640
BOOK_REFER_APPROVE_RATE = 0.45
BOOK_STRAIGHT_THROUGH_RATE = 0.97
ASSUMED_PERSONAL_LOAN_APR = 0.16
CREDIT_CARD_OBLIGATION_FRACTION = 0.05
EXISTING_DEBT_EMI_FRACTION = 0.03

# Defect counts on the default-sized files. quality.py scales them for
# smaller runs. They are counts of rows we *damage*, not a second dataset.
CUSTOMER_NULL_INCOME = 750
CUSTOMER_NULL_CITY = 500
CUSTOMER_NULL_EMPLOYMENT = 250
CUSTOMER_BAD_AGE = 40
CUSTOMER_NEGATIVE_DEBT = 40
CUSTOMER_DUPLICATE_ROWS = 80

BUREAU_DROPPED_ROWS = 40
BUREAU_NULL_SCORE = 500
BUREAU_SCORE_OUT_OF_RANGE = 60
BUREAU_BAD_UTILISATION = 65
BUREAU_DUPLICATE_ROWS = 40

APPLICATION_NULL_AMOUNT = 400
APPLICATION_NEGATIVE_AMOUNT = 60
APPLICATION_BAD_DATES = 70
APPLICATION_BAD_PRODUCT = 50
APPLICATION_BAD_CHANNEL = 30
APPLICATION_ORPHAN_CUSTOMER = 25
APPLICATION_DUPLICATE_ROWS = 40

TRANSACTION_NULL_AMOUNT = 1_000
TRANSACTION_NEGATIVE_AMOUNT = 250
TRANSACTION_BAD_DATES = 100
TRANSACTION_BLANK_COUNTRY = 40
TRANSACTION_BAD_MERCHANT = 40
TRANSACTION_NULL_FRAUD_FLAG = 80
TRANSACTION_ORPHAN_CUSTOMER = 30
TRANSACTION_DUPLICATE_ROWS = 120

PERFORMANCE_BAD_DATES = 20
PERFORMANCE_NEGATIVE_AMOUNT = 15
PERFORMANCE_ORPHAN_APPLICATION = 15
PERFORMANCE_DUPLICATE_ROWS = 20


def project_root() -> Path:
    """Folder that contains `src/` and `data/`.

    `__file__` is src/data_gen/config.py, so parents[2] is the project root.
    Using this path means the generator still writes to data/raw if you start
    it from another folder.
    """

    return Path(__file__).resolve().parents[2]


def legit_hour_weights() -> tuple[float, ...]:
    """Chance a *legitimate* transaction falls in each hour, midnight first.

    Real card spend is quiet after midnight and busier in the evening.
    Fraud patterns ignore this and place high-value tickets at night on purpose.
    """

    raw = (
        1, 1, 1, 1, 1, 2,
        4, 7, 9, 8, 7, 8,
        9, 8, 7, 7, 8, 9,
        10, 9, 7, 5, 3, 2,
    )
    total = float(sum(raw))
    return tuple(value / total for value in raw)


@dataclass(frozen=True)
class DataGenConfig:
    """Everything the generator needs, with the portfolio defaults filled in."""

    seed: int = 42
    n_customers: int = DEFAULT_N_CUSTOMERS
    n_applications: int = DEFAULT_N_APPLICATIONS
    n_transactions: int = DEFAULT_N_TRANSACTIONS
    fraud_rate: float = FRAUD_RATE
    as_of: str = AS_OF
    application_start: str = APPLICATION_START
    application_days: int = APPLICATION_DAYS
    transaction_start: str = TRANSACTION_START
    transaction_days: int = TRANSACTION_DAYS

    @property
    def n_fraud(self) -> int:
        """Fraud rows in the *clean* file, before extra duplicate rows are added."""

        return int(round(self.n_transactions * self.fraud_rate))

    def validate(self) -> None:
        """Fail early with a plain message if a practice run is too small to be useful."""

        if self.n_customers < 30:
            raise ValueError("n_customers must be at least 30.")
        if self.n_applications < 30:
            raise ValueError("n_applications must be at least 30.")
        if self.n_transactions < 100:
            raise ValueError("n_transactions must be at least 100.")
        if not 0 < self.fraud_rate < 0.05:
            raise ValueError("fraud_rate must be between 0 and 0.05 (the brief uses 0.005).")
        if abs(sum(FRAUD_SHARES.values()) - 1.0) > 1e-9:
            raise ValueError("FRAUD_SHARES must sum to 1.")
