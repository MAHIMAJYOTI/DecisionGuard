# Data lineage

Raw CSV files are loaded into DuckDB as text. dbt then types them, rejects bad rows, builds features, and publishes the tables a decision engine will read.

```mermaid
flowchart LR
    csv["data/raw CSV files"]
    raw["DuckDB raw schema<br/>all columns VARCHAR"]
    prep["prep_* checks<br/>typed rows and reject reasons"]
    stg["stg_* clean rows"]
    rej["stg_rejected_records"]
    credit["int_customer_credit_features"]
    fraud["int_transaction_fraud_features"]
    martCredit["mart_credit_applications"]
    martFraud["mart_fraud_transactions"]
    martPort["mart_portfolio_summary"]
    martDq["mart_data_quality"]

    csv --> raw
    raw --> prep
    prep --> stg
    prep --> rej
    stg --> credit
    stg --> fraud
    credit --> martCredit
    stg --> martCredit
    fraud --> martFraud
    martCredit --> martPort
    martFraud --> martPort
    rej --> martDq
```

`stg_loan_performance` joins to the credit mart for the 12-month outcome. A blank outcome means the loan is too new to judge, or it was never booked. `outcome_status` on the mart says which.

Fraud windows read only transactions that staging kept, and only transactions earlier than the current row.
