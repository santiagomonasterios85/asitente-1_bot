"""Tu asistente en Telegram.

En tu equipo:  uv run bot.py   (polling: el bot le pregunta a Telegram cada poco si hay mensajes)
En Render, el mismo archivo usa un webhook (Telegram avisa a tu URL cuando llega un mensaje):
se da cuenta solo porque Render define RENDER_EXTERNAL_URL.
"""
from io import BytesIO

from capacidades import LineaCotizacion, generar_cotizacion
from cotizador_pdf import generar_pdf

import hashlib
import os

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from asistente import responder
from config import settings

TOKEN = settings.telegram_token.get_secret_value()


async def inicio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start: lo primero que manda Telegram. Dice el número de chat para añadirlo a la lista."""
    chat = str(update.effective_chat.id)
    if chat in settings.permitidos:
        await update.message.reply_text("¡Hola! Ya puedes escribirme.")
    else:
        await update.message.reply_text(f"Tu número de chat es {chat}. Pídele al dueño del bot que lo añada.")


## Comando /cotizar: genera un PDF de cotización y lo envía a Telegram

async def cotizar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Genera y envía una cotización PDF desde Telegram."""

    chat = str(update.effective_chat.id)

    # Seguridad: solo usuarios autorizados
    if chat not in settings.permitidos:
        await update.message.reply_text(
            f"No autorizado. Tu chat es {chat}."
        )
        return

    try:
        argumentos = context.args or []

        lineas = []
        descuento = 0.0

        for argumento in argumentos:
            if argumento.lower().startswith("descuento:"):
                descuento = float(argumento.split(":", 1)[1])
                continue

            partes = argumento.split(":", 1)

            if len(partes) != 2:
                raise ValueError("Formato de producto incorrecto.")

            codigo = partes[0].strip().upper()
            cantidad = int(partes[1])

            lineas.append(
                LineaCotizacion(
                    codigo=codigo,
                    cantidad=cantidad
                )
            )

        if not lineas:
            await update.message.reply_text(
                "Uso: /cotizar L001:2 A001:3 descuento:5"
            )
            return

        # Utiliza nuestra herramienta Python existente
        resultado = generar_cotizacion(
            lineas=lineas,
            descuento_pct=descuento
        )

        if "error" in resultado:
            await update.message.reply_text(
                resultado["error"]
            )
            return

        # Crear PDF directamente en memoria
        contenido_pdf = generar_pdf(resultado)

        documento = BytesIO(contenido_pdf)
        documento.name = "cotizacion.pdf"
        documento.seek(0)

        await update.message.reply_document(
            document=documento,
            filename="cotizacion.pdf",
            caption=(
                "Cotización tecnológica generada.\n"
                f"Total: {resultado['total_bs']:,.2f} Bs\n"
                "Precios ficticios de demostración."
            )
        )

    except (ValueError, TypeError) as error:
        await update.message.reply_text(
            "No pude procesar la cotización.\n"
            "Formato: /cotizar L001:2 A001:3 descuento:5\n"
            f"Detalle: {error}"
        )

    except Exception as error:
        print("Error al generar PDF:", repr(error), flush=True)
        await update.message.reply_text(
            "Ocurrió un error al generar o enviar la cotización."
        )



async def mensaje(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = str(update.effective_chat.id)
    if chat not in settings.permitidos:  # se rechaza ANTES de llamar al modelo (y de gastar tu cuota)
        print(f"Chat no autorizado: {chat}", flush=True)
        await update.message.reply_text(f"No autorizado. Tu chat es {chat}: pídele al dueño que lo añada.")
        return
    try:
        respuesta = await responder(update.message.text)
    except Exception as error:  # cuota agotada (429), una API caída…: se avisa en vez de quedarse callado
        print("Error:", repr(error)[:300], flush=True)  # el detalle, en la terminal o en los Logs de Render
        respuesta = "Algo falló (el modelo o una herramienta). Intenta de nuevo en un minuto."
    await update.message.reply_text(respuesta[:4000])  # Telegram no acepta más de 4096 caracteres


def main() -> None:
    if not TOKEN:
        raise SystemExit("Falta TELEGRAM_TOKEN: créalo con @BotFather y ponlo en .env (o en Environment, en Render).")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", inicio))
    app.add_handler(CommandHandler("cotizar", cotizar))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, mensaje))

    permitidos = sorted(settings.permitidos) or "nadie"
    url = os.getenv("RENDER_EXTERNAL_URL")  # solo existe en Render
    if url:
        print(f"Webhook en {url}/telegram · permitidos: {permitidos}", flush=True)
        app.run_webhook(
            listen="0.0.0.0",
            port=int(os.getenv("PORT", "10000")),  # Render dice en qué puerto escuchar
            url_path="telegram",
            webhook_url=f"{url}/telegram",
            # Telegram manda este secreto en cada aviso: así nadie más puede hacerse pasar por Telegram.
            # Sale de tu token, que solo tú tienes.
            secret_token=hashlib.sha256(TOKEN.encode()).hexdigest()[:32],
        )
    else:
        print(f"Bot en marcha (polling) · permitidos: {permitidos}. Ctrl+C para parar.", flush=True)
        app.run_polling()  # ojo: esto BORRA el webhook de Render (ver README, «Problemas frecuentes»)


if __name__ == "__main__":
    main()
