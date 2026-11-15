{#
  SILVER: limpieza de negocio. Grano: una fila por reseña (review_id).
  Decisión general: NO se elimina ninguna reseña; los problemas de contenido
  se marcan con banderas para que cada análisis decida si las usa
  (ver tabla de calidad de datos del documento técnico).
#}

with bronze as (

    select * from {{ ref('brz_resenas') }}

),

limpio as (

    select
        *,
        -- Enmascara datos personales (correo, números de 10+ dígitos) y normaliza espacios.
        -- Se conservan mayúsculas y emojis (llevan señal de sentimiento).
        trim(
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        comentario,
                        '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+[.][A-Za-z]+', '[CORREO]'
                    ),
                    '[0-9]{10,}', '[NUMERO]'
                ),
                '[[:space:]]+', ' '
            )
        ) as comentario_limpio
    from bronze

),

con_palabras as (

    select
        *,
        case
            when comentario_limpio is null or comentario_limpio = '' then 0
            else array_size(split(comentario_limpio, ' '))
        end as n_palabras
    from limpio

)

select
    review_id,
    banco,
    app_id,
    comentario_limpio,
    n_palabras,
    n_palabras <= 2                                                        as es_comentario_corto,
    not coalesce(regexp_like(comentario, '.*[A-Za-zÁÉÍÓÚáéíóúÑñ].*'), false) as es_solo_simbolos,
    calificacion,
    case
        when calificacion <= 2 then 'negativo'
        when calificacion = 3  then 'neutro'
        else 'positivo'
    end                                                                    as sentimiento_estrellas,
    me_gusta,
    coalesce(version_app, 'desconocida')                                   as version_app,
    version_app is null                                                    as version_app_desconocida,
    fecha_resena,
    fecha_resena::date                                                     as fecha_dia,
    respuesta_banco,
    fecha_respuesta,
    respuesta_banco is not null                                            as tiene_respuesta_banco,
    coalesce(fecha_respuesta < fecha_resena, false)                        as respuesta_inconsistente,
    load_id,
    fecha_extraccion
from con_palabras
