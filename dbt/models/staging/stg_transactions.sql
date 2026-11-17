-- Card transactions that passed every check. One row per transaction id.
-- The fraud features are built from this set only.

select
    txn_id,
    customer_id,
    amount,
    merchant_category,
    country,
    device_id,
    txn_ts,
    is_fraud,
    _src_row
from {{ ref('prep_transactions') }}
where length(reject_reasons) = 0
qualify row_number() over (partition by txn_id order by _src_row) = 1
