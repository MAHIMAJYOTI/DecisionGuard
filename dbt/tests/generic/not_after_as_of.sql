{% test not_after_as_of(model, column_name) %}
-- Fails when a date is later than the portfolio as-of date.
select {{ column_name }} as failing_value
from {{ model }}
where cast({{ column_name }} as date) > cast('{{ var("as_of_date") }}' as date)
{% endtest %}
