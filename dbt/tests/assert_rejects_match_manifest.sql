-- Each injected defect in manifest.json must show up as the same number of
-- reject reasons. The manifest is loaded into raw.expected_issues.

with actual as (
    select
        table_name,
        reject_reason,
        count(*) as actual_rows
    from {{ ref('stg_rejected_records') }}
    group by table_name, reject_reason
),

expected as (
    select
        table_name,
        issue as reject_reason,
        rows_affected as expected_rows
    from {{ source('raw', 'expected_issues') }}
)

select
    coalesce(expected.table_name, actual.table_name) as table_name,
    coalesce(expected.reject_reason, actual.reject_reason) as reject_reason,
    expected.expected_rows,
    actual.actual_rows
from expected
full outer join actual
    on expected.table_name = actual.table_name
   and expected.reject_reason = actual.reject_reason
where coalesce(expected.expected_rows, -1) <> coalesce(actual.actual_rows, -1)
