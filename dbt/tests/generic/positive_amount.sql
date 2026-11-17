-- Generic tests. dbt calls these by the file name.
-- A test fails when this query returns any row.

{% test positive_amount(model, column_name) %}
select {{ column_name }} as failing_value
from {{ model }}
where {{ column_name }} is null
   or {{ column_name }} <= 0
{% endtest %}
