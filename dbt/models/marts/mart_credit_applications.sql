-- One row per valid application: credit features plus the 12-month outcome.
-- dpd90_within_12m is null when the outcome is not known yet (the loan is
-- too young) and also when the application was never booked. outcome_status
-- separates those two situations.

select
    features.application_id,
    features.customer_id,
    features.product,
    features.requested_amount,
    features.tenure_months,
    features.application_ts,
    features.channel,
    features.age,
    features.income,
    features.income_band,
    features.employment_type,
    features.city,
    features.existing_debt,
    features.months_at_address,
    features.bureau_score,
    features.bureau_risk_band,
    features.num_enquiries_6m,
    features.num_delinquencies_24m,
    features.total_credit_limit,
    features.utilisation,
    features.debt_to_income,
    case when performance.application_id is not null then 1 else 0 end as was_booked,
    performance.disbursed_amount,
    performance.disbursement_ts,
    performance.months_on_book,
    performance.max_dpd_12m,
    performance.dpd90_within_12m,
    case
        when performance.application_id is null then 'not_booked'
        when performance.dpd90_within_12m is null then 'not_yet_known'
        when performance.dpd90_within_12m = 1 then 'bad'
        else 'good'
    end as outcome_status
from {{ ref('int_customer_credit_features') }} as features
left join {{ ref('stg_loan_performance') }} as performance
    on features.application_id = performance.application_id
