-- Dimensión fecha. Grano: un día, desde la primera hasta la última reseña.
-- semana_inicio (lunes) coincide con la ventana semanal de carga y de alertas del PSet #1.
with limites as (

    select
        min(fecha_dia) as f_min,
        max(fecha_dia) as f_max
    from {{ ref('slv_resenas') }}

),

serie as (

    select
        dateadd(day, row_number() over (order by seq4()) - 1, l.f_min) as fecha,
        l.f_max
    from limites l,
         table(generator(rowcount => 6000))

)

select
    to_number(to_char(fecha, 'YYYYMMDD')) as fecha_key,
    fecha,
    year(fecha)                           as anio,
    quarter(fecha)                        as trimestre,
    month(fecha)                          as mes,
    to_char(fecha, 'YYYY-MM')             as anio_mes,
    date_trunc('week', fecha)::date       as semana_inicio,
    dayofweekiso(fecha)                   as dia_semana_iso
from serie
where fecha <= f_max
