-- Typed bureau row plus reject reasons. One row per raw bureau row.
-- A customer with no bureau row is not in this file; that gap is added in
-- stg_rejected_records (reason: missing_customer).

{{ config(materialized='ephemeral') }}

with source_rows as (
    select * from {{ source('raw', 'credit_bureau') }}
),

typed as (
    select
        _src_row,
        nullif(trim(customer_id), '') as customer_id,
        cast(try_cast(nullif(trim(bureau_score), '') as double) as integer) as bureau_score,
        cast(try_cast(nullif(trim(num_enquiries_6m), '') as double) as integer) as num_enquiries_6m,
        cast(try_cast(nullif(trim(num_delinquencies_24m), '') as double) as integer) as num_delinquencies_24m,
        try_cast(nullif(trim(total_credit_limit), '') as double) as total_credit_limit,
        try_cast(nullif(trim(utilisation), '') as double) as utilisation
    from source_rows
),

ranked as (
    select
        *,
        row_number() over (
            partition by customer_id
            order by _src_row
        ) as id_row_num
    from typed
)

select
    _src_row,
    customer_id,
    bureau_score,
    num_enquiries_6m,
    num_delinquencies_24m,
    total_credit_limit,
    utilisation,
    case
        when id_row_num > 1 then ['duplicate_customer_id']
        else list_filter(
            [
                case when bureau_score is null then 'null_bureau_score' end,
                case
                    when bureau_score is not null
                     and (bureau_score < 300 or bureau_score > 900)
                    then 'bureau_score_out_of_range'
                end,
                -- 1.10 means "10% over the limit" and is left alone.
                -- The injected bad values are negative or 4.5.
                case
                    when utilisation < 0 or utilisation > 1.5 then 'broken_utilisation'
                end
            ],
            x -> x is not null
        )
    end as reject_reasons
from ranked
