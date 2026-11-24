-- Dimensión banco. Grano: una fila por entidad (app de Google Play).
select
    md5(banco) as banco_key,
    banco,
    app_id
from (
    select distinct banco, app_id
    from {{ ref('slv_resenas') }}
)
