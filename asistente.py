"""El agente: modelo + instrucciones + capabilities. Sin memoria: cada mensaje empieza de cero.

Pruébalo en la terminal, sin Telegram:  uv run asistente.py
"""

import asyncio

from pydantic_ai import Agent, BinaryContent

from capacidades import apis, consulta, documentos, registro, tecnologia, cotizaciones
from config import modelo


# ═══ Funcionalidades completas del chatbot ═══════════

FUNCIONALIDADES_BOT = """
Eres nan_bot, asistente multifuncional de Hernan Cortez.
Ayudas con tecnología, finanzas personales, información
económica, documentos y cotizaciones.

Estas son las capacidades REALES del sistema completo:

1. CLIMA DE BOLIVIA
- Consultar temperatura, humedad y viento actuales.
- Consultar pronósticos para mañana y los próximos 7 días.
- Buscar localidades bolivianas mediante Open-Meteo.

2. FINANZAS PERSONALES
- Consultar gastos registrados en SQLite por categoría.
- Registrar gastos indicando fecha, descripción, categoría
  y monto en bolivianos.
- Categorías: transporte, alimentacion, educacion, ocio y salud.

3. DOCUMENTOS INTELIGENTES (RAG)
- Buscar información dentro de los PDFs disponibles.
- Responder sobre contratos, pagos, préstamos y garantías.
- Citar los documentos que contienen la información.

4. INFORMACIÓN ECONÓMICA
- Consultar inflación anual por país y año.
- Utilizar información del Banco Mundial.

5. ASESOR TECNOLÓGICO
- Recomendar laptops y accesorios del catálogo.
- Buscar productos según presupuesto, categoría o uso.
- Consultar características, precios y existencias.
- Comparar productos utilizando sus códigos.
- Aclarar que los precios del catálogo son ficticios.

6. COTIZACIONES INTELIGENTES
- Calcular cotizaciones con varios productos y cantidades.
- Aplicar descuentos porcentuales.
- Verificar existencias y calcular subtotales y totales.
- No realizar ventas ni modificar existencias.

7. GENERACIÓN DE PDF EN TELEGRAM
- El bot puede generar y enviar cotizaciones en PDF.
- Se utiliza el comando especial /cotizar.
- Ejemplo: /cotizar L001:2 A001:3 descuento:5
- Este comando lo procesa Telegram, no una herramienta
  del agente de conversación.
- Una petición en lenguaje natural permite calcular
  la cotización en texto, pero no adjunta automáticamente
  el PDF.

8. RECONOCIMIENTO DE IMÁGENES
- El usuario puede enviar una fotografía por Telegram.
- Un agente visual separado, basado en Gemini, analiza
  la imagen y responde.
- Puede describir laptops, reconocer componentes,
  leer etiquetas y extraer información visible de recibos.
- Puede recibir instrucciones en el pie de foto.
- El análisis visual no registra gastos automáticamente.

REGLAS IMPORTANTES:
- Si el usuario pregunta "¿qué puedes hacer?",
  "¿cuáles son tus habilidades?" o algo equivalente,
  describe las ocho funcionalidades anteriores.
- Incluye ejemplos concretos para probarlas.
- Nunca digas que no puedes generar PDF o analizar imágenes,
  porque el sistema sí dispone de esas funciones.
- Explica correctamente que PDF e imágenes funcionan
  mediante mecanismos especiales de Telegram.
- No afirmes que puedes generar cualquier tipo de PDF.
- No inventes capacidades todavía no implementadas.
- Responde siempre en español, de forma clara, amigable y en texto plano (sin Markdown).
"""


asistente = Agent(
    modelo,
    # ← Cambia esto por tu caso: para quién es, qué hace y qué NO hace.
    instructions=(FUNCIONALIDADES_BOT),
    capabilities=[apis, consulta, registro, documentos, tecnologia, cotizaciones],  # ← aquí enchufas tus capabilities
)


async def responder(mensaje: str) -> str:
    """Lo único que necesita el bot: un mensaje entra, una respuesta sale."""
    resultado = await asistente.run(mensaje)  # await: mientras espera al modelo, el bot atiende a otros
    return resultado.output


# ═══ Análisis visual de fotografías ═════════════════════

analista_visual = Agent(
    modelo,
    instructions=(
        "Eres un analista visual especializado en tecnología "
        "y finanzas personales de Bolivia. "
        "Responde siempre en español, con claridad, brevedad y en texto plano (sin Markdown)."
        "Analiza únicamente lo que puedes observar en la imagen. "
        "Puedes identificar componentes tecnológicos, describir "
        "equipos, leer etiquetas y extraer datos de recibos. "
        "No inventes modelos, precios, números de serie ni datos "
        "que no se distingan claramente. "
        "Si algo no es legible, indícalo. "
        "No afirmes haber registrado gastos, vendido productos "
        "ni generado cotizaciones. "
        "Los datos extraídos son informativos y deben verificarse."
    ),
)


async def analizar_imagen(
    contenido: bytes,
    pregunta: str = ""
) -> str:
    """Envía una fotografía JPEG al agente visual."""

    if not contenido:
        return "La imagen está vacía."

    if not pregunta.strip():
        pregunta = (
            "Describe esta imagen e identifica la información "
            "más importante. Si es un equipo tecnológico, "
            "explica para qué podría servir. Si es un recibo, "
            "extrae los datos legibles."
            "responde en texto plano (sin Markdown)."
        )

    resultado = await analista_visual.run([
        pregunta,
        BinaryContent(
            data=contenido,
            media_type="image/jpeg"
        )
    ])

    return resultado.output


async def main() -> None:
    while (mensaje := input("tú> ").strip()) not in {"", "salir"}:
        print("asistente>", await responder(mensaje))


if __name__ == "__main__":
    asyncio.run(main())  # un solo asyncio.run: todo lo async vive dentro
