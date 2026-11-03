
"""Genera cotizaciones PDF desde resultados validados."""

from datetime import datetime
from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


def generar_pdf(cotizacion: dict) -> bytes:
    """Recibe una cotización válida y devuelve su PDF en memoria."""

    if cotizacion.get("estado") != "cotizacion_generada":
        raise ValueError("No se puede generar un PDF de una cotización inválida.")

    buffer = BytesIO()

    documento = SimpleDocTemplate(
        buffer,
        pagesize=(21 * cm, 29.7 * cm),
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    estilos = getSampleStyleSheet()

    titulo = ParagraphStyle(
        "TituloCotizacion",
        parent=estilos["Title"],
        fontSize=20,
        textColor=colors.HexColor("#17365D"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )

    normal = estilos["Normal"]

    derecha = ParagraphStyle(
        "DerechaCotizacion",
        parent=normal,
        alignment=TA_RIGHT,
    )

    elementos = []

    elementos.append(
        Paragraph("BOLIVIA TECH", titulo)
    )
    elementos.append(
        Paragraph(
            "Cotización tecnológica de demostración",
            estilos["Heading2"]
        )
    )

    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")

    elementos.append(
        Paragraph(f"Fecha de emisión: {fecha}", normal)
    )

    elementos.append(
        Paragraph(
            "Moneda: Bolivianos (BOB)",
            normal
        )
    )

    elementos.append(Spacer(1, 0.6 * cm))

    encabezados = [
        "Código", "Producto", "Cant.", "P. Unit.", "Subtotal"
    ]

    filas = [encabezados]

    for producto in cotizacion["productos"]:
        filas.append([
            escape(str(producto["codigo"])),
            Paragraph(
                escape(str(producto["producto"])),
                normal
            ),
            str(producto["cantidad"]),
            f'{producto["precio_unitario_bs"]:,.2f}',
            f'{producto["subtotal_bs"]:,.2f}',
        ])

    tabla = Table(
        filas,
        colWidths=[2 * cm, 6 * cm, 1.5 * cm, 3 * cm, 3 * cm],
        repeatRows=1,
        hAlign="CENTER",
    )

    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F0F4F8")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
    ]))

    elementos.append(tabla)
    elementos.append(Spacer(1, 0.8 * cm))

    elementos.append(Paragraph(
        f'Subtotal: {cotizacion["subtotal_bs"]:,.2f} Bs',
        derecha
    ))

    elementos.append(Paragraph(
        f'Descuento ({cotizacion["descuento_pct"]:g}%): '
        f'{cotizacion["descuento_bs"]:,.2f} Bs',
        derecha
    ))

    elementos.append(Spacer(1, 0.3 * cm))

    elementos.append(Paragraph(
        f'<b>TOTAL: {cotizacion["total_bs"]:,.2f} Bs</b>',
        ParagraphStyle(
            "TotalCotizacion",
            parent=derecha,
            fontSize=16,
            textColor=colors.HexColor("#17365D"),
        )
    ))

    elementos.append(Spacer(1, 1 * cm))

    elementos.append(Paragraph(
        "Documento informativo generado automáticamente. "
        "Los precios y existencias son ficticios. "
        "No constituye una venta ni reserva de productos.",
        normal
    ))

    documento.build(elementos)

    return buffer.getvalue()
