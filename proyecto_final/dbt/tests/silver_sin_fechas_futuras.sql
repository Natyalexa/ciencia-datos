-- Validez: una reseña no puede tener fecha posterior al momento de la carga.
select review_id, fecha_resena, fecha_extraccion
from {{ ref('slv_resenas') }}
where fecha_resena > fecha_extraccion
