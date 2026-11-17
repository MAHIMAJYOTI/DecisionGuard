-- Bureau rows that passed every check. One row per customer.

select
    customer_id,
    bureau_score,
    num_enquiries_6m,
    num_delinquencies_24m,
    total_credit_limit,
    utilisation,
    _src_row
from {{ ref('prep_credit_bureau') }}
where length(reject_reasons) = 0
qualify row_number() over (partition by customer_id order by _src_row) = 1
