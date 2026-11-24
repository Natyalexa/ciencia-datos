-- Regla de negocio: la estrella no puede perder ni duplicar reseñas de ningún banco.
-- Devuelve filas (y falla) si el conteo por banco en Gold difiere del de Silver.
with silver as (
    select banco, count(*) as n from {{ ref('slv_resenas') }} group by banco
),
gold as (
    select d.banco, count(*) as n
    from {{ ref('fct_resena') }} f
    join {{ ref('dim_banco') }} d on f.banco_key = d.banco_key
    group by d.banco
)
select
    coalesce(s.banco, g.banco) as banco,
    s.n as filas_silver,
    g.n as filas_gold
from silver s
full outer join gold g on s.banco = g.banco
where s.n is distinct from g.n
