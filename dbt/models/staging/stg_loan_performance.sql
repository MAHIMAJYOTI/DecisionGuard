-- Booked loans that passed every check. A blank dpd90_within_12m stays blank.

select
    application_id,
    customer_id,
    disbursed_amount,
    disbursement_ts,
    months_on_book,
    max_dpd_12m,
    dpd90_within_12m,
    _src_row
from {{ ref('prep_loan_performance') }}
where length(reject_reasons) = 0
qualify row_number() over (partition by application_id order by _src_row) = 1
