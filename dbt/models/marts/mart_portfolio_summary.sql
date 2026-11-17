-- Daily and monthly book totals.
-- Booking rate uses the application date. Fraud rate uses the transaction date.
-- A day with applications but no card spend keeps fraud_rate null, rather
-- than pretending the fraud rate was zero before the card file starts.
--
-- COUNT(*) FILTER (WHERE ...) is DuckDB / PostgreSQL.
-- BigQuery writes the same idea as COUNTIF(was_booked = 1).

with application_days as (
    select
        cast(date_trunc('day', application_ts) as date) as period_start,
        count(*) as application_count,
        count(*) filter (where was_booked = 1) as booked_count,
        sum(requested_amount) as requested_amount,
        sum(disbursed_amount) as disbursed_amount
    from {{ ref('mart_credit_applications') }}
    group by 1
),

transaction_days as (
    select
        cast(date_trunc('day', txn_ts) as date) as period_start,
        count(*) as transaction_count,
        sum(is_fraud) as fraud_count,
        sum(amount) as transaction_amount
    from {{ ref('mart_fraud_transactions') }}
    group by 1
),

application_months as (
    select
        cast(date_trunc('month', application_ts) as date) as period_start,
        count(*) as application_count,
        count(*) filter (where was_booked = 1) as booked_count,
        sum(requested_amount) as requested_amount,
        sum(disbursed_amount) as disbursed_amount
    from {{ ref('mart_credit_applications') }}
    group by 1
),

transaction_months as (
    select
        cast(date_trunc('month', txn_ts) as date) as period_start,
        count(*) as transaction_count,
        sum(is_fraud) as fraud_count,
        sum(amount) as transaction_amount
    from {{ ref('mart_fraud_transactions') }}
    group by 1
),

stacked as (
    select
        'day' as period_grain,
        coalesce(applications.period_start, transactions.period_start) as period_start,
        applications.application_count,
        applications.booked_count,
        applications.booked_count * 1.0 / nullif(applications.application_count, 0) as booking_rate,
        applications.requested_amount,
        applications.disbursed_amount,
        transactions.transaction_count,
        transactions.fraud_count,
        transactions.fraud_count * 1.0 / nullif(transactions.transaction_count, 0) as fraud_rate,
        transactions.transaction_amount
    from application_days as applications
    full outer join transaction_days as transactions
        on applications.period_start = transactions.period_start

    union all

    select
        'month' as period_grain,
        coalesce(applications.period_start, transactions.period_start) as period_start,
        applications.application_count,
        applications.booked_count,
        applications.booked_count * 1.0 / nullif(applications.application_count, 0) as booking_rate,
        applications.requested_amount,
        applications.disbursed_amount,
        transactions.transaction_count,
        transactions.fraud_count,
        transactions.fraud_count * 1.0 / nullif(transactions.transaction_count, 0) as fraud_rate,
        transactions.transaction_amount
    from application_months as applications
    full outer join transaction_months as transactions
        on applications.period_start = transactions.period_start
)

select
    period_grain,
    period_start,
    application_count,
    booked_count,
    booking_rate,
    requested_amount,
    disbursed_amount,
    transaction_count,
    fraud_count,
    fraud_rate,
    transaction_amount
from stacked
