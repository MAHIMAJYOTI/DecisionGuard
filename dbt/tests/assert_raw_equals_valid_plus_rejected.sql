-- Valid staging rows + distinct rejected source rows must equal the raw file.
-- missing_customer has no source row (the bureau line was deleted), so those
-- rejects are excluded here and checked against the manifest instead.

with raw_counts as (
    select 'customers' as table_name, count(*) as raw_rows from {{ source('raw', 'customers') }}
    union all
    select 'credit_bureau', count(*) from {{ source('raw', 'credit_bureau') }}
    union all
    select 'loan_applications', count(*) from {{ source('raw', 'loan_applications') }}
    union all
    select 'transactions', count(*) from {{ source('raw', 'transactions') }}
    union all
    select 'loan_performance', count(*) from {{ source('raw', 'loan_performance') }}
),

valid_counts as (
    select 'customers' as table_name, count(*) as valid_rows from {{ ref('stg_customers') }}
    union all
    select 'credit_bureau', count(*) from {{ ref('stg_credit_bureau') }}
    union all
    select 'loan_applications', count(*) from {{ ref('stg_loan_applications') }}
    union all
    select 'transactions', count(*) from {{ ref('stg_transactions') }}
    union all
    select 'loan_performance', count(*) from {{ ref('stg_loan_performance') }}
),

rejected_counts as (
    select table_name, count(distinct _src_row) as rejected_rows
    from {{ ref('stg_rejected_records') }}
    where _src_row is not null
    group by table_name
)

select
    raw_counts.table_name,
    raw_counts.raw_rows,
    valid_counts.valid_rows,
    coalesce(rejected_counts.rejected_rows, 0) as rejected_rows
from raw_counts
inner join valid_counts
    on raw_counts.table_name = valid_counts.table_name
left join rejected_counts
    on raw_counts.table_name = rejected_counts.table_name
where raw_counts.raw_rows
    <> valid_counts.valid_rows + coalesce(rejected_counts.rejected_rows, 0)
