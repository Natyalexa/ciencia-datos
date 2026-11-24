"""Operaciones sobre el Model Registry: estado, promoción manual y rollback (sección 4).

    python -m src.registro_modelos estado
    python -m src.registro_modelos rollback                 # vuelve a champion_anterior
    python -m src.registro_modelos rollback --a-version 1   # vuelve a una versión concreta
    python -m src.registro_modelos promover --version 2

El rollback solo mueve el alias: la imagen de la API no se reconstruye. La API consulta el alias
cada RECARGA_SEG segundos (o al llamar a POST /admin/recargar) y GET /modelo refleja el cambio.
"""
from __future__ import annotations

import argparse
import os

from src import config as C


def cliente():
    import mlflow
    mlflow.set_tracking_uri(C.MLFLOW_URI)
    return mlflow.MlflowClient()


def estado():
    cli = cliente()
    rm = cli.get_registered_model(C.NOMBRE_MODELO)
    print(f"Modelo: {C.NOMBRE_MODELO}")
    print("Alias:", {k: v for k, v in rm.aliases.items()})
    for mv in sorted(cli.search_model_versions(f"name='{C.NOMBRE_MODELO}'"), key=lambda m: int(m.version)):
        met = cli.get_run(mv.run_id).data.metrics
        print(f"  v{mv.version} run_id={mv.run_id} fase={mv.tags.get('fase', cli.get_run(mv.run_id).data.tags.get('fase'))} "
              f"decision={mv.tags.get('decision', '-')} umbral={met.get('umbral_aplicado')}")


def mover(version: str, alias: str = C.ALIAS_PRODUCCION, motivo: str = ""):
    cli = cliente()
    actual = cli.get_model_version_by_alias(C.NOMBRE_MODELO, C.ALIAS_PRODUCCION).version
    if alias == C.ALIAS_PRODUCCION and str(actual) != str(version):
        cli.set_registered_model_alias(C.NOMBRE_MODELO, "champion_anterior", str(actual))
    cli.set_registered_model_alias(C.NOMBRE_MODELO, alias, str(version))
    if motivo:
        cli.set_model_version_tag(C.NOMBRE_MODELO, str(version), "ultimo_movimiento", motivo)
    print(f"Alias {alias}: v{actual} → v{version}")
    _avisar_api()


def rollback(a_version: str | None):
    cli = cliente()
    destino = a_version or cli.get_model_version_by_alias(C.NOMBRE_MODELO, "champion_anterior").version
    mover(str(destino), motivo="rollback")


def _avisar_api():
    """Opcional: pide a la API recargar ya (si no, lo hace sola en ≤ RECARGA_SEG segundos)."""
    try:
        import requests
        r = requests.post(os.environ.get("API_URL", "http://api:8000") + "/admin/recargar",
                          headers={"X-Token-Admin": os.environ.get("ADMIN_TOKEN", "")}, timeout=60)
        if r.ok:
            print("API recargada →", r.json())
    except Exception:  # noqa: BLE001
        print("La API se actualizará sola en el próximo ciclo de revisión del alias.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("estado")
    rb = sub.add_parser("rollback")
    rb.add_argument("--a-version")
    pr = sub.add_parser("promover")
    pr.add_argument("--version", required=True)
    a = ap.parse_args()
    {"estado": estado, "rollback": lambda: rollback(a.a_version),
     "promover": lambda: mover(a.version, motivo="promoción manual")}[a.cmd]()
