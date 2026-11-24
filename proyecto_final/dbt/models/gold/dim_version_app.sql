-- Dimensión versión de la app. Grano: una versión DE UN BANCO.
-- La versión se identifica junto con el banco porque el mismo número (por ejemplo 5.2.0)
-- puede existir en apps distintas y no significa lo mismo.
select
    md5(banco || '|' || version_app) as version_app_key,
    md5(banco)                       as banco_key,
    banco,
    version_app,
    version_app = 'desconocida'      as es_desconocida,
    min(fecha_dia)                   as primera_resena,
    max(fecha_dia)                   as ultima_resena
from {{ ref('slv_resenas') }}
group by banco, version_app
