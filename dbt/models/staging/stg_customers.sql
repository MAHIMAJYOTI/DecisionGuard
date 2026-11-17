-- Customers that passed every check. QUALIFY keeps the first file row per id.
-- BigQuery supports QUALIFY as well. Rows that fail a check are not dropped
-- on the floor: prep_customers lists them, and stg_rejected_records stores them.

select
    customer_id,
    age,
    income,
    employment_type,
    city,
    existing_debt,
    months_at_address,
    _src_row
from {{ ref('prep_customers') }}
where length(reject_reasons) = 0
qualify row_number() over (partition by customer_id order by _src_row) = 1
