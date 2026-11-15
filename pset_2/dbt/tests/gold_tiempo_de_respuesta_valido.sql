-- Regla de negocio: un banco no puede responder antes de que exista la reseña.
-- Las respuestas inconsistentes quedan sin tiempo de respuesta (null); nunca negativo.
select review_id, horas_hasta_respuesta
from {{ ref('fct_resena') }}
where horas_hasta_respuesta < 0
