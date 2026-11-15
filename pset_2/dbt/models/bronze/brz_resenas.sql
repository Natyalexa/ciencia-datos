{#
  BRONZE: copia fiel y tipada de RAW, con un único ajuste técnico:
  se elimina el review_id duplicado que genera la carga incremental
  (se solapa a propósito la última reseña ya cargada).
  Regla: se conserva la fila de la extracción más reciente.
#}

with fuente as (

    select * from {{ source('raw', 'resenas_google_play') }}

),

ordenadas as (

    select
        *,
        row_number() over (
            partition by review_id
            order by extracted_at desc, load_id desc
        ) as rn
    from fuente

)

select
    review_id,
    banco,
    app_id,
    content                 as comentario,
    score                   as calificacion,
    thumbs_up_count         as me_gusta,
    review_created_version  as version_app_creacion,
    app_version             as version_app,
    review_at               as fecha_resena,
    reply_content           as respuesta_banco,
    replied_at              as fecha_respuesta,
    load_id,
    extracted_at            as fecha_extraccion
from ordenadas
where rn = 1
