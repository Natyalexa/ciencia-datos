{#
  TABLA DE HECHOS. Grano: UNA RESEÑA (review_id).
  El par (reseña, aspecto) del PSet #1 no se modela aún: las etiquetas de aspecto
  se construirán con anotación manual; cuando existan, se agrega fct_resena_aspecto
  con este mismo review_id como llave.
  comentario_limpio y sentimiento_estrellas son atributos degenerados (sin dimensión propia).
#}
select
    review_id,
    md5(banco)                                         as banco_key,
    to_number(to_char(fecha_dia, 'YYYYMMDD'))          as fecha_key,
    md5(banco || '|' || version_app)                   as version_app_key,

    calificacion,
    sentimiento_estrellas,
    me_gusta,
    n_palabras,
    es_comentario_corto,
    es_solo_simbolos,
    tiene_respuesta_banco,
    respuesta_inconsistente,
    case
        when tiene_respuesta_banco and not respuesta_inconsistente
            then datediff(hour, fecha_resena, fecha_respuesta)
    end                                                as horas_hasta_respuesta,

    comentario_limpio,
    fecha_resena
from {{ ref('slv_resenas') }}
