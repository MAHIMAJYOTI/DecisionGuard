-- One typed customer row, plus the list of reasons it should be rejected.
-- Ephemeral means dbt pastes this query into the models that use it.
-- It is not stored as its own table.
--
-- TRY_CAST is DuckDB. BigQuery's name for the same function is SAFE_CAST.
-- A value that is not a real number becomes NULL instead of failing the query.

{{ config(materialized='ephemeral') }}

with source_rows as (
    select * from {{ source('raw', 'customers') }}
),

typed as (
    select
        _src_row,
        nullif(trim(customer_id), '') as customer_id,
        cast(try_cast(nullif(trim(age), '') as double) as integer) as age,
        try_cast(nullif(trim(income), '') as double) as income,
        nullif(lower(trim(employment_type)), '') as employment_type,
        nullif(trim(city), '') as city,
        try_cast(nullif(trim(existing_debt), '') as double) as existing_debt,
        cast(try_cast(nullif(trim(months_at_address), '') as double) as integer) as months_at_address
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
    age,
    income,
    employment_type,
    city,
    existing_debt,
    months_at_address,
    case
        -- A later copy of the same id is only a duplicate. Its other bad
        -- values are echoes of the first copy, and the manifest counted the
        -- extra row once, as a duplicate.
        when id_row_num > 1 then ['duplicate_customer_id']
        else list_filter(
            [
                case when income is null then 'null_income' end,
                case when city is null then 'null_city' end,
                case when employment_type is null then 'null_employment_type' end,
                case when age is null or age < 18 or age > 100 then 'impossible_age' end,
                case when existing_debt < 0 then 'negative_existing_debt' end
            ],
            -- DuckDB lambda. Drops the NULL slots so the list only holds real reasons.
            -- BigQuery would use a different list function; the idea is the same.
            x -> x is not null
        )
    end as reject_reasons
from ranked
