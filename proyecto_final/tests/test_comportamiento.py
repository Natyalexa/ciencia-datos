"""3. Comportamiento de negocio: propiedades que el modelo debe cumplir en este dominio."""
import pytest

from api.servicio import ServicioModelo

QUEJA = " no funciona, pesimo servicio, siempre se cae"
NEUTRAS = [
    "hice una transferencia hoy en la tarde",
    "actualice la aplicacion esta semana",
    "uso la app para pagar la tarjeta",
    "revise mi saldo en la manana",
]
CLARAS = [
    ("excelente aplicacion, muy rapida y facil de usar", "cola_normal"),
    ("muy buena app, me encanta, todo funciona bien", "cola_normal"),
    ("pesima app, no me deja ingresar y se cuelga siempre", "priorizar"),
    ("horrible, no funciona desde la actualizacion, no puedo transferir", "priorizar"),
]


def _p(modelo, texto, banco="Banco Pichincha"):
    X = ServicioModelo.preparar({"comentario_limpio": texto, "banco": banco,
                                 "es_solo_simbolos": 0, "version_app_desconocida": 0})
    return float(modelo.predict_proba(X)[:, 1][0])


@pytest.mark.parametrize("texto", NEUTRAS)
def test_agregar_una_queja_no_reduce_la_probabilidad(modelo, texto):
    """Prueba direccional: añadir una queja explícita a una reseña nunca la vuelve menos negativa."""
    assert _p(modelo, texto + QUEJA) >= _p(modelo, texto)


@pytest.mark.parametrize("texto,esperado", CLARAS)
def test_casos_inequivocos(modelo, metadata, texto, esperado):
    """Funcionalidad mínima: elogios claros no se priorizan y quejas claras sí."""
    p = _p(modelo, texto)
    decision = "priorizar" if p >= metadata["umbral"] else "cola_normal"
    assert decision == esperado, f"p={p:.3f}"


@pytest.mark.parametrize("banco", ["Banco Guayaquil", "Banco del Pacífico", "Cooperativa JEP"])
def test_queja_clara_se_prioriza_en_cualquier_banco(modelo, metadata, banco):
    """Equidad mínima entre entidades: una queja inequívoca se prioriza sin importar el banco."""
    assert _p(modelo, "pesima app, no me deja ingresar y se cuelga siempre", banco) >= metadata["umbral"]
