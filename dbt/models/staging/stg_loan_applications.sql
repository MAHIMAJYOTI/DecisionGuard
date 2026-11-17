-- Applications that passed every check. One row per application id.

select
    application_id,
    customer_id,
    product,
    requested_amount,
    tenure_months,
    application_ts,
    channel,
    _src_row
from {{ ref('prep_loan_applications') }}
where length(reject_reasons) = 0
qualify row_number() over (partition by application_id order by _src_row) = 1
