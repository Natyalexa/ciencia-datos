-- Regla de privacidad (compromiso del PSet #1): no puede quedar ningún correo
-- ni número de 10+ dígitos (teléfono, cédula, tarjeta) en el texto de Silver.
select review_id, comentario_limpio
from {{ ref('slv_resenas') }}
where regexp_like(comentario_limpio, '.*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+[.][A-Za-z]+.*')
   or regexp_like(comentario_limpio, '.*[0-9]{10,}.*')
