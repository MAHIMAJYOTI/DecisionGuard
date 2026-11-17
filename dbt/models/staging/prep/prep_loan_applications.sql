-- Typed application row plus reject reasons.
-- Product is NOT lower-cased before the check. 'CREDIT CARD' is a different
-- code from 'credit card'; fixing the case quietly would hide 13 injected
-- defects and the manifest count would no longer match.

{{ config(materialized='ephemeral') }}

with source_rows as (
    select * from {{ source('raw', 'loan_applications') }}
),

known_customers as (
    select distinct nullif(trim(customer_id), '') as customer_id
    from {{ source('raw', 'customers') }}
),

typed as (
    select
        source_rows._src_row,
        nullif(trim(source_rows.application_id), '') as application_id,
        nullif(trim(source_rows.customer_id), '') as customer_id,
        nullif(trim(source_rows.product), '') as product,
        try_cast(nullif(trim(source_rows.requested_amount), '') as double) as requested_amount,
        cast(try_cast(nullif(trim(source_rows.tenure_months), '') as double) as integer) as tenure_months,
        try_cast(nullif(trim(source_rows.application_ts), '') as timestamp) as application_ts,
        nullif(lower(trim(source_rows.channel)), '') as channel,
        known_customers.customer_id as matched_customer_id
    from source_rows
    left join known_customers
        on known_customers.customer_id = nullif(trim(source_rows.customer_id), '')
),

ranked as (
    select
        *,
        row_number() over (
            partition by application_id
            order by _src_row
        ) as id_row_num
    from typed
)

select
    _src_row,
    application_id,
    customer_id,
    product,
    requested_amount,
    tenure_months,
    application_ts,
    channel,
    case
        when id_row_num > 1 then ['duplicate_application_id']
        else list_filter(
            [
                case when requested_amount is null then 'null_requested_amount' end,
                case when requested_amount < 0 then 'negative_requested_amount' end,
                case
                    when application_ts is null
                      or year(application_ts) < 1990
                      or cast(application_ts as date) > cast('{{ var("as_of_date") }}' as date)
                    then 'bad_application_ts'
                end,
                case
                    when product is null
                      or product not in ('personal loan', 'credit card')
                    then 'bad_product'
                end,
                case
                    when channel is null
                      or channel not in ('branch', 'web', 'app')
                    then 'bad_channel'
                end,
                case
                    when matched_customer_id is null then 'orphan_customer_id'
                end
            ],
            x -> x is not null
        )
    end as reject_reasons
from ranked
