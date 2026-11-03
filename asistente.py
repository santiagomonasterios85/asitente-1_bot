"""El agente: modelo + instrucciones + capabilities. Sin memoria: cada mensaje empieza de cero.

Pruébalo en la terminal, sin Telegram:  uv run asistente.py
"""

import asyncio

from pydantic_ai import Agent, BinaryContent

from capacidades import apis, consulta, documentos, registro, tecnologia, cotizaciones
from config import modelo

asistente = Agent(
    modelo,
    # ← Cambia esto por tu caso: para quién es, qué hace y qué NO hace.
    instructions=(
        "Eres el asistente personal de Hernan Cortez: economía, clima, sus gastos y sus documentos. "
        "Responde en español, breve y en texto plano (sin Markdown)."
    ),
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
        "Responde siempre en español, con claridad y brevedad. "
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
