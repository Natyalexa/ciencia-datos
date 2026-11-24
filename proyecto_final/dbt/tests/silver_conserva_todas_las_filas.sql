-- Regla de negocio: Silver marca los problemas pero no descarta reseñas.
-- Falla (devuelve una fila) si Silver tiene distinto número de filas que Bronze.
select
    (select count(*) from {{ ref('brz_resenas') }}) as filas_bronze,
    (select count(*) from {{ ref('slv_resenas') }}) as filas_silver
where (select count(*) from {{ ref('brz_resenas') }})
   != (select count(*) from {{ ref('slv_resenas') }})
