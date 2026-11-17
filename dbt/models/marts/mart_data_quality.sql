-- How many rows were rejected, by source table and reason.
-- This is the table the monitoring dashboard will read later.

select
    table_name,
    reject_reason,
    count(*) as rejected_rows
from {{ ref('stg_rejected_records') }}
group by table_name, reject_reason
