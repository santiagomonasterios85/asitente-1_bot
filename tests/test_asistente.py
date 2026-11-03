"""Pruebas sin modelo real ni internet: uv run pytest

Un modelo falso (FunctionModel) pide la tool que queremos, y comprobamos lo que hace el código.
"""

import os
from types import SimpleNamespace

os.environ.setdefault("GEMINI_API_KEY_LOCAL", "clave-de-prueba")  # para importar sin .env (no se llama a Gemini)

import httpx  # noqa: E402
import pytest  # noqa: E402
from pydantic import SecretStr  # noqa: E402
from pydantic_ai import ModelResponse, TextPart, ToolCallPart, ToolReturnPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

import api  # noqa: E402
import asistente  # noqa: E402
import bot  # noqa: E402
import capacidades  # noqa: E402
from config import settings  # noqa: E402


@pytest.fixture(autouse=True)
def base_temporal(tmp_path, monkeypatch):
    """Cada prueba usa su propia gastos.db, no la de datos/."""
    monkeypatch.setattr(capacidades, "DB", tmp_path / "gastos.db")


def pide(tool: str, args: dict) -> FunctionModel:
    """Un modelo falso: la 1.ª vez pide la tool; cuando recibe su resultado, responde texto."""

    def responder(mensajes, info):
        ultimo = mensajes[-1].parts[-1]
        if isinstance(ultimo, ToolReturnPart):
            return ModelResponse(parts=[TextPart(f"Resultado: {ultimo.content}")])
        return ModelResponse(parts=[ToolCallPart(tool, args)])

    return FunctionModel(responder)


def test_consultar_gastos_sin_modelo():
    gastos = capacidades.consultar_gastos("transporte")  # una tool es una función normal
    assert gastos["total_bs"] == 67


async def test_registrar_gasto_escribe_en_la_base():
    with asistente.asistente.override(model=pide("registrar_gasto", {
        "fecha": "2030-05-14", "categoria": "ocio", "descripcion": "Teatro", "monto": 30,
    })):
        respuesta = await asistente.responder("Anota 30 Bs de teatro.")
    assert "Gasto registrado" in respuesta
    assert capacidades.consultar_gastos("ocio")["total_bs"] == 70  # 40 del CSV + 30


async def test_monto_negativo_no_se_guarda():
    with asistente.asistente.override(model=pide("registrar_gasto", {
        "fecha": "2030-05-14", "categoria": "ocio", "descripcion": "Error", "monto": -5,
    })):
        respuesta = await asistente.responder("Anota -5 Bs.")
    assert "No se guardó" in respuesta
    assert capacidades.consultar_gastos("ocio")["total_bs"] == 40


async def test_chat_ajeno_se_rechaza_sin_llamar_al_modelo(monkeypatch):
    monkeypatch.setattr(settings, "telegram_permitidos", "123")
    respuestas = []

    async def reply_text(texto):
        respuestas.append(texto)

    def modelo_prohibido(mensajes, info):
        raise AssertionError("no debía llamarse al modelo")

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=999), message=SimpleNamespace(text="hola", reply_text=reply_text)
    )
    with asistente.asistente.override(model=FunctionModel(modelo_prohibido)):
        await bot.mensaje(update, None)
    assert "No autorizado" in respuestas[0] and "999" in respuestas[0]


def test_permitidos(monkeypatch):
    monkeypatch.setattr(settings, "telegram_permitidos", " 123, 456 ,")
    assert settings.permitidos == {"123", "456"}
    monkeypatch.setattr(settings, "telegram_permitidos", "")
    assert settings.permitidos == set()  # vacío = nadie


# ── Extra opcional: la API (API.md) ──


async def post_chat(cuerpo: dict, token: str | None = None) -> httpx.Response:
    """Llama a la API en el mismo proceso, sin servidor ni red."""
    cabeceras = {"Authorization": f"Bearer {token}"} if token else {}
    transporte = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transporte, base_url="http://prueba") as cliente:
        return await cliente.post("/chat", json=cuerpo, headers=cabeceras)


async def test_api_sin_token_valido_no_llama_al_modelo(monkeypatch):
    def modelo_prohibido(mensajes, info):
        raise AssertionError("no debía llamarse al modelo")

    with asistente.asistente.override(model=FunctionModel(modelo_prohibido)):
        monkeypatch.setattr(settings, "api_tokens", SecretStr(""))
        assert (await post_chat({"mensaje": "hola"}, "cualquiera")).status_code == 401  # vacío = nadie
        monkeypatch.setattr(settings, "api_tokens", SecretStr("abc, def"))
        assert (await post_chat({"mensaje": "hola"})).status_code == 401
        assert (await post_chat({"mensaje": "hola"}, "xyz")).status_code == 401
        assert (await post_chat({"mensaje": ""}, "abc")).status_code == 422  # Pydantic valida el pedido


async def test_api_con_token_responde(monkeypatch):
    monkeypatch.setattr(settings, "api_tokens", SecretStr("abc,def"))
    with asistente.asistente.override(model=pide("consultar_gastos", {"categoria": "transporte"})):
        r = await post_chat({"mensaje": "¿Cuánto gasté en transporte?"}, "def")
    assert r.status_code == 200
    assert "67" in r.json()["texto"]







def test_clima_bolivia_manana(monkeypatch):
    """Comprueba el clima sin utilizar internet."""

    class RespuestaFalsa:
        def __init__(self, datos):
            self.datos = datos

        def raise_for_status(self):
            pass

        def json(self):
            return self.datos

    def api_falsa(url, params, timeout):
        if "geocoding-api" in url:
            return RespuestaFalsa({
                "results": [{
                    "name": "La Paz",
                    "admin1": "La Paz",
                    "latitude": -16.5,
                    "longitude": -68.15
                }]
            })

        return RespuestaFalsa({
            "current": {
                "temperature_2m": 12,
                "relative_humidity_2m": 60
            },
            "daily": {
                "time": ["2026-11-02", "2026-11-03"],
                "temperature_2m_max": [18, 20],
                "temperature_2m_min": [5, 6],
                "precipitation_probability_max": [20, 70],
                "precipitation_sum": [0, 3]
            }
        })

    monkeypatch.setattr(capacidades.httpx, "get", api_falsa)

    resultado = capacidades.clima_bolivia("La Paz", "manana")

    assert resultado["ciudad"] == "La Paz"
    assert resultado["pronostico"]["fecha"] == "2026-11-03"
    assert resultado["pronostico"]["maxima_c"] == 20
    assert resultado["pronostico"]["probabilidad_lluvia_pct"] == 70


# ═══ Pruebas del asesor tecnológico ═══════════════════

def test_buscar_laptops_por_presupuesto():
    resultado = capacidades.buscar_productos(
        presupuesto_max=5500,
        categoria="Laptop"
    )

    assert resultado["total_encontrados"] > 0

    for producto in resultado["productos"]:
        assert producto["precio_bs"] <= 5500
        assert producto["categoria"] == "Laptop"
        assert producto["stock"] > 0


def test_detalle_producto():
    producto = capacidades.detalle_producto("L001")

    assert producto["codigo"] == "L001"
    assert producto["marca"] == "Lenovo"
    assert producto["precio_bs"] == 5200


def test_comparar_productos():
    resultado = capacidades.comparar_productos("L001", "L004")

    assert resultado["producto_1"]["codigo"] == "L001"
    assert resultado["producto_2"]["codigo"] == "L004"
    assert resultado["diferencia_precio_bs"] == 1700


# ═══ Pruebas del cotizador tecnológico ═══════════════════

def test_cotizacion_con_descuento():
    from capacidades import LineaCotizacion

    resultado = capacidades.generar_cotizacion(
        lineas=[
            LineaCotizacion(codigo="L001", cantidad=2),
            LineaCotizacion(codigo="A001", cantidad=3)
        ],
        descuento_pct=5
    )

    assert resultado["subtotal_bs"] == 10760
    assert resultado["descuento_bs"] == 538
    assert resultado["total_bs"] == 10222


def test_cotizacion_stock_insuficiente():
    from capacidades import LineaCotizacion

    resultado = capacidades.generar_cotizacion(
        lineas=[
            LineaCotizacion(codigo="L005", cantidad=10)
        ]
    )

    assert "error" in resultado
    assert resultado["disponible"] == 2


def test_cotizacion_producto_inexistente():
    from capacidades import LineaCotizacion

    resultado = capacidades.generar_cotizacion(
        lineas=[
            LineaCotizacion(codigo="NOEXISTE", cantidad=1)
        ]
    )

    assert "error" in resultado


async def test_telegram_envia_cotizacion_pdf(monkeypatch):
    """Comprueba el envío de PDF sin llamar a Telegram."""

    monkeypatch.setattr(
        settings, "telegram_permitidos", "123"
    )

    documentos_enviados = []

    async def reply_text(texto):
        raise AssertionError(
            f"No se esperaba un mensaje de error: {texto}"
        )

    async def reply_document(document, filename, caption):
        documentos_enviados.append({
            "contenido": document.read(),
            "nombre": filename,
            "caption": caption
        })

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=123),
        message=SimpleNamespace(
            reply_text=reply_text,
            reply_document=reply_document
        )
    )

    context = SimpleNamespace(
        args=["L001:2", "A001:3", "descuento:5"]
    )

    await bot.cotizar(update, context)

    assert len(documentos_enviados) == 1

    enviado = documentos_enviados[0]

    assert enviado["contenido"].startswith(b"%PDF-")
    assert enviado["nombre"] == "cotizacion.pdf"
    assert "10,222.00 Bs" in enviado["caption"]



# ═══ Pruebas del análisis de imágenes ═══════════════

async def test_analizar_imagen_con_modelo_falso():
    """Comprueba el agente visual sin utilizar Gemini."""

    from pydantic_ai import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    def respuesta_falsa(mensajes, info):
        return ModelResponse(
            parts=[TextPart(
                "Es una laptop para programación."
            )]
        )

    modelo_falso = FunctionModel(respuesta_falsa)

    with asistente.analista_visual.override(
        model=modelo_falso
    ):
        respuesta = await asistente.analizar_imagen(
            contenido=b"\xff\xd8\xffimagen-de-prueba",
            pregunta="¿Qué observas?"
        )

    assert "laptop" in respuesta.lower()


async def test_foto_usuario_no_autorizado(monkeypatch):
    """Un usuario ajeno no puede consumir Gemini."""

    monkeypatch.setattr(
        settings, "telegram_permitidos", "123"
    )

    respuestas = []

    async def reply_text(texto):
        respuestas.append(texto)

    async def modelo_prohibido(*args, **kwargs):
        raise AssertionError(
            "No se debería llamar al análisis visual."
        )

    monkeypatch.setattr(
        bot, "analizar_imagen", modelo_prohibido
    )

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=999),
        message=SimpleNamespace(
            photo=[],
            reply_text=reply_text
        )
    )

    await bot.recibir_foto(update, None)

    assert "No autorizado" in respuestas[0]
