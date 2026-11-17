-- One row per valid application, with the customer and bureau features used
-- for a credit decision.
--
-- Debt-to-income uses the same EMI formula as src/data_gen/performance.py:
--   existing monthly payment = existing_debt * 0.03
--   personal loan payment    = standard annuity at 16% APR
--   credit card payment      = 5% of the requested limit
--   debt_to_income           = (existing + new) / (income / 12)
-- When income is zero the simulator stores 99, meaning "cannot afford this".
-- The formula is recomputed here from the raw inputs. It is not copied from
-- the performance file.

{% set monthly_rate = var('personal_loan_apr') / 12.0 %}

with applications as (
    select * from {{ ref('stg_loan_applications') }}
),

customers as (
    select * from {{ ref('stg_customers') }}
),

bureau as (
    select * from {{ ref('stg_credit_bureau') }}
),

joined as (
    select
        applications.application_id,
        applications.customer_id,
        applications.product,
        applications.requested_amount,
        applications.tenure_months,
        applications.application_ts,
        applications.channel,
        customers.age,
        customers.income,
        customers.employment_type,
        customers.city,
        customers.existing_debt,
        customers.months_at_address,
        bureau.bureau_score,
        bureau.num_enquiries_6m,
        bureau.num_delinquencies_24m,
        bureau.total_credit_limit,
        bureau.utilisation,
        customers.existing_debt * {{ var('existing_debt_emi_fraction') }} as existing_emi,
        case
            when applications.product = 'personal loan'
             and applications.tenure_months >= 1
                then applications.requested_amount
                    * {{ monthly_rate }}
                    * pow(1 + {{ monthly_rate }}, applications.tenure_months)
                    / (pow(1 + {{ monthly_rate }}, applications.tenure_months) - 1)
            when applications.product = 'credit card'
                then applications.requested_amount * {{ var('credit_card_obligation_fraction') }}
        end as new_monthly_obligation
    from applications
    left join customers
        on applications.customer_id = customers.customer_id
    left join bureau
        on applications.customer_id = bureau.customer_id
)

select
    application_id,
    customer_id,
    product,
    requested_amount,
    tenure_months,
    application_ts,
    channel,
    age,
    income,
    employment_type,
    city,
    existing_debt,
    months_at_address,
    bureau_score,
    num_enquiries_6m,
    num_delinquencies_24m,
    total_credit_limit,
    utilisation,
    case
        when bureau_score is null then null
        when bureau_score < 600 then 'high'
        when bureau_score < 700 then 'medium'
        when bureau_score < 800 then 'low'
        else 'very_low'
    end as bureau_risk_band,
    case
        when income is null then null
        when income <= 0 then 'no_income'
        when income < 300000 then 'under_3_lakh'
        when income < 600000 then '3_to_6_lakh'
        when income < 1200000 then '6_to_12_lakh'
        else 'over_12_lakh'
    end as income_band,
    existing_emi,
    new_monthly_obligation,
    case
        when income is null or existing_debt is null or new_monthly_obligation is null then null
        when income <= 0 then 99.0
        else (existing_emi + new_monthly_obligation) / (income / 12.0)
    end as debt_to_income
from joined
