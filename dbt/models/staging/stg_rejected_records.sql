-- Every rejected raw row, one line per reason.
-- A customer with a blank income and a bad age appears twice.
-- missing_customer has no _src_row: that person was deleted from the bureau
-- file, so there is no bureau line to point at. The reconciliation test
-- ignores those rows when it checks "valid + rejected = raw".

with stacked as (
    select
        'customers' as table_name,
        customer_id as record_id,
        _src_row,
        reject_reasons
    from {{ ref('prep_customers') }}
    where length(reject_reasons) > 0

    union all

    select
        'credit_bureau',
        customer_id,
        _src_row,
        reject_reasons
    from {{ ref('prep_credit_bureau') }}
    where length(reject_reasons) > 0

    union all

    select
        'loan_applications',
        application_id,
        _src_row,
        reject_reasons
    from {{ ref('prep_loan_applications') }}
    where length(reject_reasons) > 0

    union all

    select
        'transactions',
        txn_id,
        _src_row,
        reject_reasons
    from {{ ref('prep_transactions') }}
    where length(reject_reasons) > 0

    union all

    select
        'loan_performance',
        application_id,
        _src_row,
        reject_reasons
    from {{ ref('prep_loan_performance') }}
    where length(reject_reasons) > 0
),

exploded as (
    select
        stacked.table_name,
        stacked.record_id,
        reason.reject_reason,
        stacked._src_row
    from stacked
    cross join unnest(stacked.reject_reasons) as reason(reject_reason)
),

missing_bureau as (
    select
        'credit_bureau' as table_name,
        customers.customer_id as record_id,
        'missing_customer' as reject_reason,
        cast(null as bigint) as _src_row
    from (
        select distinct nullif(trim(customer_id), '') as customer_id
        from {{ source('raw', 'customers') }}
    ) as customers
    left join (
        select distinct nullif(trim(customer_id), '') as customer_id
        from {{ source('raw', 'credit_bureau') }}
    ) as bureau
        on customers.customer_id = bureau.customer_id
    where bureau.customer_id is null
      and customers.customer_id is not null
)

select table_name, record_id, reject_reason, _src_row
from exploded
where reject_reason is not null

union all

select table_name, record_id, reject_reason, _src_row
from missing_bureau
