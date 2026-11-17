-- Typed performance row plus reject reasons.
-- A blank dpd90_within_12m is kept. It means the 12-month outcome is not
-- known yet, which is not a data-quality error.
--
-- APP00000000 is the injected orphan key. Several different loans were
-- overwritten with that same id, so sharing the id does not make them
-- duplicates. A duplicate is an exact copy of an earlier row. Each distinct
-- orphan loan can still carry its own bad date or negative amount.

{{ config(materialized='ephemeral') }}

with source_rows as (
    select * from {{ source('raw', 'loan_performance') }}
),

known_applications as (
    select distinct nullif(trim(application_id), '') as application_id
    from {{ source('raw', 'loan_applications') }}
),

typed as (
    select
        source_rows._src_row,
        nullif(trim(source_rows.application_id), '') as application_id,
        nullif(trim(source_rows.customer_id), '') as customer_id,
        try_cast(nullif(trim(source_rows.disbursed_amount), '') as double) as disbursed_amount,
        try_cast(nullif(trim(source_rows.disbursement_ts), '') as timestamp) as disbursement_ts,
        cast(try_cast(nullif(trim(source_rows.months_on_book), '') as double) as integer) as months_on_book,
        cast(try_cast(nullif(trim(source_rows.max_dpd_12m), '') as double) as integer) as max_dpd_12m,
        cast(try_cast(nullif(trim(source_rows.dpd90_within_12m), '') as double) as integer) as dpd90_within_12m,
        known_applications.application_id as matched_application_id
    from source_rows
    left join known_applications
        on known_applications.application_id = nullif(trim(source_rows.application_id), '')
),

-- Most duplicated ids are an original row plus an exact copy appended later.
-- APP00000000 is different: several different loans were overwritten with
-- that same id, so they are not copies of each other. An exact copy is the
-- only row that matches an earlier row on every business column. That copy
-- is the duplicate. Each distinct orphan loan stays on its own and can still
-- carry a bad date or a negative amount.
ranked as (
    select
        *,
        row_number() over (
            partition by
                application_id,
                customer_id,
                disbursed_amount,
                disbursement_ts,
                months_on_book,
                max_dpd_12m,
                dpd90_within_12m
            order by _src_row
        ) as id_row_num
    from typed
)

select
    _src_row,
    application_id,
    customer_id,
    disbursed_amount,
    disbursement_ts,
    months_on_book,
    max_dpd_12m,
    dpd90_within_12m,
    case
        when id_row_num > 1 then ['duplicate_application_id']
        else list_filter(
            [
                case when matched_application_id is null then 'orphan_application_id' end,
                case
                    when disbursement_ts is null
                      or year(disbursement_ts) < 1990
                      or cast(disbursement_ts as date) > cast('{{ var("as_of_date") }}' as date)
                    then 'bad_disbursement_ts'
                end,
                case when disbursed_amount < 0 then 'negative_disbursed_amount' end
            ],
            x -> x is not null
        )
    end as reject_reasons
from ranked
