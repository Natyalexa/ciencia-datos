{# Usa el nombre de esquema tal cual (BRONZE, SILVER, GOLD) en lugar de <esquema_base>_<nombre>. #}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema | trim }}
    {%- else -%}
        {{ custom_schema_name | trim | upper }}
    {%- endif -%}
{%- endmacro %}
