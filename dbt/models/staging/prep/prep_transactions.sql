-- Typed card transaction plus reject reasons.
-- Only rows that survive this list are allowed onto the fraud timeline.
-- A negative amount or a broken timestamp would distort the 1-hour window.

{{ config(materialized='ephemeral') }}

with source_rows as (
    select * from {{ source('raw', 'transactions') }}
),

known_customers as (
    select distinct nullif(trim(customer_id), '') as customer_id
    from {{ source('raw', 'customers') }}
),

typed as (
    select
        source_rows._src_row,
        nullif(trim(source_rows.txn_id), '') as txn_id,
        nullif(trim(source_rows.customer_id), '') as customer_id,
        try_cast(nullif(trim(source_rows.amount), '') as double) as amount,
        nullif(lower(trim(source_rows.merchant_category)), '') as merchant_category,
        nullif(upper(trim(source_rows.country)), '') as country,
        nullif(trim(source_rows.device_id), '') as device_id,
        try_cast(nullif(trim(source_rows.txn_ts), '') as timestamp) as txn_ts,
        cast(try_cast(nullif(trim(source_rows.is_fraud), '') as double) as integer) as is_fraud,
        known_customers.customer_id as matched_customer_id
    from source_rows
    left join known_customers
        on known_customers.customer_id = nullif(trim(source_rows.customer_id), '')
),

ranked as (
    select
        *,
        row_number() over (
            partition by txn_id
            order by _src_row
        ) as id_row_num
    from typed
)

select
    _src_row,
    txn_id,
    customer_id,
    amount,
    merchant_category,
    country,
    device_id,
    txn_ts,
    is_fraud,
    case
        when id_row_num > 1 then ['duplicate_txn_id']
        else list_filter(
            [
                case when amount is null then 'null_amount' end,
                case when amount < 0 then 'negative_amount' end,
                case
                    when txn_ts is null
                      or year(txn_ts) < 1990
                      or cast(txn_ts as date) > cast('{{ var("as_of_date") }}' as date)
                    then 'bad_txn_ts'
                end,
                case when country is null then 'blank_country' end,
                case
                    when merchant_category is null
                      or merchant_category not in (
                          'grocery', 'fuel', 'restaurant', 'ecommerce', 'travel',
                          'atm', 'electronics', 'utilities', 'jewellery', 'healthcare'
                      )
                    then 'bad_merchant_category'
                end,
                case
                    when is_fraud is null or is_fraud not in (0, 1) then 'null_is_fraud'
                end,
                case when matched_customer_id is null then 'orphan_customer_id' end
            ],
            x -> x is not null
        )
    end as reject_reasons
from ranked
