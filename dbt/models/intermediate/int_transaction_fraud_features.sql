-- One row per valid card transaction, with point-in-time fraud features.
--
-- A feature may only use transactions strictly earlier than the current row.
-- If the current purchase were included, one huge ticket would pull its own
-- average up and the "5 times average" spike would look milder than it is.
--
-- DuckDB accepts a RANGE frame written with INTERVAL on the timestamp:
--   RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND INTERVAL 1 MILLISECOND PRECEDING
-- The millisecond bound drops the current timestamp, so the current row (and
-- any other row in the same second) is excluded.
--
-- BigQuery equivalent, ordering by UNIX seconds instead of the timestamp:
--   SUM(amount) OVER (
--     PARTITION BY customer_id
--     ORDER BY UNIX_SECONDS(txn_ts)
--     RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING
--   )
-- "1 PRECEDING" drops the current second. Use 86400 for 24 hours and
-- 2592000 for 30 days.

with txns as (
    select * from {{ ref('stg_transactions') }}
),

windowed as (
    select
        txn_id,
        customer_id,
        amount,
        merchant_category,
        country,
        device_id,
        txn_ts,
        is_fraud,
        coalesce(count(*) over (
            partition by customer_id
            order by txn_ts
            range between interval 1 hour preceding and interval 1 millisecond preceding
        ), 0) as txn_count_1h,
        coalesce(sum(amount) over (
            partition by customer_id
            order by txn_ts
            range between interval 1 hour preceding and interval 1 millisecond preceding
        ), 0) as txn_amount_1h,
        coalesce(count(*) over (
            partition by customer_id
            order by txn_ts
            range between interval 24 hour preceding and interval 1 millisecond preceding
        ), 0) as txn_count_24h,
        coalesce(sum(amount) over (
            partition by customer_id
            order by txn_ts
            range between interval 24 hour preceding and interval 1 millisecond preceding
        ), 0) as txn_amount_24h,
        avg(amount) over (
            partition by customer_id
            order by txn_ts
            range between interval 30 day preceding and interval 1 millisecond preceding
        ) as avg_amount_30d,
        -- First time this device appears for this customer. txn_id breaks ties
        -- when two rows share a timestamp, so exactly one of them is "new".
        case
            when row_number() over (
                partition by customer_id, device_id
                order by txn_ts, txn_id
            ) = 1 then 1
            else 0
        end as is_new_device,
        case when country <> 'IN' then 1 else 0 end as is_foreign_country,
        -- 00:00 up to but not including 05:00. 05:00 itself is morning.
        case when extract(hour from txn_ts) < 5 then 1 else 0 end as is_night
    from txns
)

select
    txn_id,
    customer_id,
    amount,
    merchant_category,
    country,
    device_id,
    txn_ts,
    is_fraud,
    txn_count_1h,
    txn_amount_1h,
    txn_count_24h,
    txn_amount_24h,
    avg_amount_30d,
    amount / nullif(avg_amount_30d, 0) as amount_vs_avg_30d,
    is_new_device,
    is_foreign_country,
    is_night
from windowed
