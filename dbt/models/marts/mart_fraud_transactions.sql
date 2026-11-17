-- One row per valid card transaction, with the fraud features and the label.

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
    amount_vs_avg_30d,
    is_new_device,
    is_foreign_country,
    is_night
from {{ ref('int_transaction_fraud_features') }}
