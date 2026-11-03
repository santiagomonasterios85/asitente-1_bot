"""Las capabilities de tu asistente: lo que vimos en clase. Cámbialas o agrega las tuyas.

Una Capability junta tools + instrucciones en una pieza que se enchufa al agente (asistente.py).
El docstring y los tipos de cada tool son lo único que el modelo lee para decidir cuándo usarla.
"""

import csv
import sqlite3
from datetime import date
from functools import cache
from typing import Literal

from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, Field

import chromadb
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic_ai.capabilities import Capability
from pypdf import PdfReader

from config import DATOS, settings

# ═══ S1 · Tools de lectura con APIs públicas ══════════════════════════════════════════
'''apis = Capability(
    id="apis_publicas",
    description="Consulta indicadores económicos (Banco Mundial) y clima (Open Meteo) con sus APIs públicas",
    instructions="Usa las APIs para obtener datos reales. Si una API no tiene el dato, dilo, y no inventes datos",
)'''


apis = Capability(
    id="apis_publicas",
    description="Consulta inflación del Banco Mundial y clima actual o pronósticos de ciudades de Bolivia mediante Open-Meteo",
    instructions=(
        "Usa las APIs para obtener datos reales y actualizados. "
        "Para consultas del clima en Bolivia utiliza clima_bolivia. "
        "Si preguntan por mañana usa periodo='manana'. "
        "Si preguntan por los próximos 7 días usa periodo='semana'. "
        "No inventes temperaturas ni pronósticos."
    ),
)

@apis.tool_plain
def inflacion(pais: str, anio: int) -> dict:
    """Inflación anual de un país en un año, según el Banco Mundial.

    Args:
        pais: código ISO del país, por ejemplo 'BO' para Bolivia o 'PE' para Perú
        anio: año, por ejemplo 2024
    """
    url = f"https://api.worldbank.org/v2/country/{pais}/indicator/FP.CPI.TOTL.ZG"
    respuesta = httpx.get(url, params={"date": anio, "format": "json"}, timeout=10)
    respuesta.raise_for_status()
    datos = respuesta.json()

    # La API responde [metadatos, registros]. Si no hay dato, devolvemos un aviso que el modelo puede leer
    if len(datos) < 2 or not datos[1] or datos[1][0]["value"] is None:
        return {"error": f"No hay datos de inflación disponibles para {pais} en {anio}"}

    registro = datos[1][0]
    return {
        "pais": registro["country"]["value"],
        "anio": registro["date"],
        "inflacion_pct": round(registro["value"], 2),
        "fuente": "Banco Mundial",  # devolver la fuente evita que el modelo invente una
    }


@apis.tool_plain
def clima_bolivia(
    ciudad: str,
    periodo: Literal["hoy", "manana", "semana"] = "hoy"
) -> dict:
    """Consulta el clima actual o pronóstico de una ciudad de Bolivia.

    Args:
        ciudad: Nombre de la ciudad boliviana, por ejemplo La Paz,
                El Alto, Cochabamba, Santa Cruz, Sucre o Tarija.
        periodo: hoy, manana (mañana) o semana (7 días).
    """
    try:
        # 1. Buscar coordenadas de la ciudad
        ubicacion = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": ciudad,
                "count": 10,
                "language": "es",
                "countryCode": "BO"
            },
            timeout=15
        )
        ubicacion.raise_for_status()
        resultados = ubicacion.json().get("results", [])

        if not resultados:
            return {"error": f"No encontré la localidad '{ciudad}' en Bolivia."}

        lugar = resultados[0]

        # 2. Consultar el pronóstico del tiempo
        respuesta = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lugar["latitude"],
                "longitude": lugar["longitude"],
                "current": (
                    "temperature_2m,relative_humidity_2m,"
                    "apparent_temperature,wind_speed_10m"
                ),
                "daily": (
                    "temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max,precipitation_sum"
                ),
                "timezone": "America/La_Paz",
                "forecast_days": 7
            },
            timeout=15
        )
        respuesta.raise_for_status()
        datos = respuesta.json()

        info = {
            "ciudad": lugar["name"],
            "departamento": lugar.get("admin1", "No especificado"),
            "pais": "Bolivia",
            "fuente": "Open-Meteo",
            "periodo": periodo
        }

        if periodo == "hoy":
            info["actual"] = datos["current"]
            indice = 0
        elif periodo == "manana":
            indice = 1
        else:
            indice = None

        diario = datos["daily"]

        def pronostico(i):
            return {
                "fecha": diario["time"][i],
                "maxima_c": diario["temperature_2m_max"][i],
                "minima_c": diario["temperature_2m_min"][i],
                "probabilidad_lluvia_pct":
                    diario["precipitation_probability_max"][i],
                "precipitacion_mm": diario["precipitation_sum"][i]
            }

        if indice is not None:
            info["pronostico"] = pronostico(indice)
        else:
            info["pronostico"] = [
                pronostico(i) for i in range(len(diario["time"]))
            ]

        return info

    except (httpx.HTTPError, ValueError, KeyError, IndexError) as error:
        return {
            "error": f"No pude consultar el clima de {ciudad}.",
            "detalle": str(error)
        }

'''@apis.tool_plain
def clima_cochabamba() -> dict:
    """Temperatura actual de Cochabamba, según Open Meteo."""
    params = {"latitude": -17.38, "longitude": -66.15, "current": "temperature_2m"}
    respuesta = httpx.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
    respuesta.raise_for_status()
    return {"temperatura": respuesta.json()["current"]["temperature_2m"], "fuente": "Open Meteo"}'''


# ═══ S1 · Mi contador: gastos en SQLite, una tool que lee y otra que escribe ══════════
Categoria = Literal["transporte", "alimentacion", "educacion", "ocio", "salud"]
CSV = DATOS / "gastos.csv"  # 10 gastos ficticios de mayo de 2030
DB = DATOS / "gastos.db"  # la base: se crea desde el CSV la primera vez


def crear_base() -> None:
    """Crea gastos.db desde cero con los gastos del CSV."""
    DB.unlink(missing_ok=True)
    with CSV.open(encoding="utf-8") as archivo:
        filas = [(g["fecha"], g["categoria"], g["descripcion"], float(g["monto"])) for g in csv.DictReader(archivo)]
    conexion = sqlite3.connect(DB)
    with conexion:
        conexion.execute(
            "CREATE TABLE gastos (fecha TEXT, categoria TEXT, descripcion TEXT, monto REAL CHECK (monto > 0))"
        )
        conexion.executemany("INSERT INTO gastos VALUES (?, ?, ?, ?)", filas)
    conexion.close()


def hoy() -> str:
    """Instrucción dinámica: el modelo no sabe qué día es; se lo decimos en cada mensaje."""
    return f"Hoy es {date.today().isoformat()}."


# Lectura y escritura van en capabilities separadas: a otro agente le podrías dar solo la primera
consulta = Capability(
    id="consulta_gastos",
    description="Consulta gastos personales almacenados en una base de datos SQLite",
    instructions="Usa la base de datos para responder preguntas sobre los gastos. No inventes datos.",
)
registro = Capability(
    id="registro_gastos",
    description="Registra nuevos gastos en la base de datos SQLite",
    instructions=["Agrega nuevos gastos a la base de datos. Asegúrate de que los datos sean válidos.", hoy],
)


@consulta.tool_plain
def consultar_gastos(categoria: Categoria) -> dict:
    """Gastos de una categoría y su total en bolivianos."""
    if not DB.exists():
        crear_base()
    conexion = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)  # solo lectura
    conexion.row_factory = sqlite3.Row  # cada fila como diccionario
    filas = conexion.execute(
        "SELECT fecha, descripcion, monto FROM gastos WHERE categoria = ?",  # SQL fijo; el valor va en ?
        (categoria,),
    ).fetchall()
    conexion.close()
    return {"categoria": categoria, "total_bs": sum(f["monto"] for f in filas), "gastos": [dict(f) for f in filas]}


@registro.tool_plain
def registrar_gasto(fecha: str, categoria: Categoria, descripcion: str, monto: float) -> str:
    """Anota un gasto nuevo. La fecha va como AAAA-MM-DD y el monto en bolivianos."""
    if not DB.exists():
        crear_base()
    conexion = sqlite3.connect(DB)  # sin mode=ro: esta tool sí escribe
    try:
        with conexion:  # transacción: guarda todo o nada
            conexion.execute("INSERT INTO gastos VALUES (?, ?, ?, ?)", (fecha, categoria, descripcion, monto))
    except sqlite3.IntegrityError:  # el CHECK de la tabla rechaza montos de 0 o negativos
        return f"No se guardó: el monto debe ser mayor que 0 (recibí {monto})."
    finally:
        conexion.close()
    return f"Gasto registrado: {descripcion}, {monto} Bs en {categoria} el {fecha}."


# ═══ S2 · RAG: buscar en mis documentos (PDFs) y citar la fuente ══════════════════════
DOCUMENTOS = DATOS / "documentos"  # PDFs ficticios de finanzas personales


def embeddings_de(textos: list[str]) -> list[list[float]]:
    """Convierte cada texto en un vector: textos de significado parecido dan vectores cercanos."""
    # Gemini también ofrece una API compatible con la de OpenAI; la usamos para los embeddings
    cliente = OpenAI(
        api_key=settings.gemini_api_key_local.get_secret_value(),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    respuesta = cliente.embeddings.create(model="gemini-embedding-001", input=textos)
    return [item.embedding for item in respuesta.data]


@cache  # se indexa la primera vez que alguien busca, y una sola vez mientras el programa corre
def indice():
    # Base vectorial en memoria (se pierde al terminar); "cosine" mide la cercanía por el ángulo
    coleccion = chromadb.EphemeralClient().get_or_create_collection(
        "documentos", configuration={"hnsw": {"space": "cosine"}}
    )
    for pdf in sorted(DOCUMENTOS.glob("*.pdf")):
        texto = "\n".join(pagina.extract_text() for pagina in PdfReader(pdf).pages)
        # Trozos de ~300 caracteres que se solapan 50, para no cortar una idea por la mitad
        trozos = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_text(texto)
        coleccion.upsert(
            ids=[f"{pdf.stem}-{i}" for i in range(len(trozos))],  # el id dice de qué PDF sale: sirve para citar
            documents=trozos,
            embeddings=embeddings_de(trozos),
        )
    return coleccion


def buscar(pregunta: str, k: int = 3) -> list[dict]:
    """Los k trozos más parecidos a la pregunta, con su fuente y su similitud (1 = idéntico)."""
    r = indice().query(query_embeddings=embeddings_de([pregunta]), n_results=k)
    return [
        {"fuente": fuente, "similitud": round(1 - distancia, 2), "texto": texto}
        for fuente, distancia, texto in zip(r["ids"][0], r["distances"][0], r["documents"][0])
    ]


documentos = Capability(
    id="mis_documentos",
    description="Documentos de finanzas personales: contratos, facturas y garantías",
    # Citar la fuente y admitir cuando no hay datos: así se ve qué sale de los documentos y qué no
    instructions=(
        "Para preguntas sobre pagos, contratos o garantías usa buscar_documentos y cita la fuente "
        "entre corchetes, por ejemplo [contrato_alquiler-1]. "
        "Si no hay información suficiente para responder, di que no lo encuentras, no lo inventes."
    ),
)


@documentos.tool_plain
def buscar_documentos(pregunta: str) -> list[dict]:
    """Busca en los documentos los fragmentos que responden a la pregunta."""
    resultados = buscar(pregunta, k=3)
    for r in resultados:  # en la terminal (o en los Logs de Render) se ve qué recuperó
        print(f"  📄 {r['fuente']} (similitud {r['similitud']})", flush=True)
    return resultados







# ═══ S3 · Asesor tecnológico y catálogo de productos ═══════════

CATALOGO = DATOS / "catalogo.csv"

tecnologia = Capability(
    id="asesor_tecnologico",
    description=(
        "Consulta un catálogo de laptops y accesorios tecnológicos. "
        "Recomienda productos por presupuesto, características y uso."
    ),
    instructions=(
        "Utiliza las herramientas del catálogo para recomendar productos. "
        "Los precios están expresados en bolivianos (Bs). "
        "No inventes productos, precios, existencias ni características. "
        "El catálogo contiene datos ficticios de demostración. "
        "Para recomendaciones explica brevemente por qué cada equipo "
        "es adecuado para el uso solicitado."
    ),
)


def cargar_catalogo() -> list[dict]:
    """Lee los productos del archivo CSV."""
    with CATALOGO.open(encoding="utf-8-sig", newline="") as archivo:
        productos = list(csv.DictReader(archivo))

    for producto in productos:
        producto["precio_bs"] = float(producto["precio_bs"])
        producto["ram_gb"] = int(producto["ram_gb"])
        producto["almacenamiento_gb"] = int(producto["almacenamiento_gb"])
        producto["stock"] = int(producto["stock"])

    return productos


@tecnologia.tool_plain
def buscar_productos(
    presupuesto_max: float = 0,
    categoria: str = "",
    uso: str = ""
) -> dict:
    """Busca productos disponibles por presupuesto, categoría o uso.

    Args:
        presupuesto_max: Presupuesto máximo en bolivianos. 0 significa sin límite.
        categoria: Laptop o Accesorio. Vacío significa todas.
        uso: programacion, gaming, oficina, estudios, diseno,
             almacenamiento o actualizacion. Vacío significa todos.
    """
    productos = cargar_catalogo()
    resultados = []

    for producto in productos:
        if producto["stock"] <= 0:
            continue

        if presupuesto_max > 0 and producto["precio_bs"] > presupuesto_max:
            continue

        if categoria and producto["categoria"].lower() != categoria.lower():
            continue

        if uso and producto["uso"].lower() != uso.lower():
            continue

        resultados.append(producto)

    resultados.sort(key=lambda p: p["precio_bs"])

    return {
        "total_encontrados": len(resultados),
        "productos": resultados,
        "fuente": "Catálogo ficticio de demostración"
    }


@tecnologia.tool_plain
def detalle_producto(codigo: str) -> dict:
    """Obtiene las características, precio y stock de un producto por código."""
    for producto in cargar_catalogo():
        if producto["codigo"].upper() == codigo.upper():
            return producto

    return {"error": f"No existe el producto con código {codigo}."}


@tecnologia.tool_plain
def comparar_productos(codigo1: str, codigo2: str) -> dict:
    """Compara dos productos del catálogo utilizando sus códigos."""
    productos = cargar_catalogo()

    encontrados = {
        p["codigo"].upper(): p
        for p in productos
    }

    primero = encontrados.get(codigo1.upper())
    segundo = encontrados.get(codigo2.upper())

    if primero is None or segundo is None:
        return {"error": "Uno o ambos productos no existen en el catálogo."}

    return {
        "producto_1": primero,
        "producto_2": segundo,
        "diferencia_precio_bs": round(
            abs(primero["precio_bs"] - segundo["precio_bs"]), 2
        ),
        "fuente": "Catálogo ficticio de demostración"
    }


# ═══ Siguiente módulo · Cotizaciones tecnológicas ═══════════

class LineaCotizacion(BaseModel):
    """Un producto solicitado y su cantidad."""
    codigo: str
    cantidad: int = Field(gt=0)


cotizaciones = Capability(
    id="cotizaciones_tecnologicas",
    description=(
        "Genera cotizaciones de laptops y accesorios usando "
        "los precios y existencias del catálogo."
    ),
    instructions=(
        "Para cotizar productos utiliza generar_cotizacion. "
        "Obtén los códigos de los productos del catálogo. "
        "El descuento se expresa como porcentaje: 5 significa 5%. "
        "No inventes precios ni realices cálculos por tu cuenta. "
        "Si falta un producto o no hay stock, informa del problema. "
        "Los precios son ficticios y se expresan en bolivianos. "
        "No afirmes que se ha realizado una venta."
    ),
)


def dinero(valor: Decimal) -> Decimal:
    """Redondea cantidades monetarias a dos decimales."""
    return valor.quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP
    )


@cotizaciones.tool_plain
def generar_cotizacion(
    lineas: list[LineaCotizacion],
    descuento_pct: float = 0
) -> dict:
    """Cotiza productos del catálogo y comprueba existencias.

    Args:
        lineas: Lista de productos con código y cantidad.
        descuento_pct: Descuento porcentual entre 0 y 100.
                        Por ejemplo, 5 significa 5%.
    """
    descuento = Decimal(str(descuento_pct))

    if not descuento.is_finite() or not 0 <= descuento <= 100:
        return {"error": "El descuento debe estar entre 0 y 100%."}

    if not lineas:
        return {"error": "La cotización necesita productos."}

    catalogo = {
        p["codigo"].upper(): p
        for p in cargar_catalogo()
    }

    # Agrupar códigos repetidos para comprobar stock total.
    cantidades = {}

    for linea in lineas:
        codigo = linea.codigo.strip().upper()
        if not codigo:
            return {"error": "Hay un código de producto vacío."}

        cantidades[codigo] = (
            cantidades.get(codigo, 0) + linea.cantidad
        )

    # Validar todo antes de calcular.
    for codigo, cantidad in cantidades.items():
        producto = catalogo.get(codigo)

        if producto is None:
            return {"error": f"Producto {codigo} no encontrado."}

        if cantidad > producto["stock"]:
            return {
                "error": f"Stock insuficiente para {codigo}.",
                "solicitado": cantidad,
                "disponible": producto["stock"]
            }

    detalle = []
    subtotal = Decimal("0")

    for codigo, cantidad in cantidades.items():
        producto = catalogo[codigo]
        precio = dinero(Decimal(str(producto["precio_bs"])))
        importe = dinero(precio * cantidad)
        subtotal += importe

        detalle.append({
            "codigo": codigo,
            "producto": (
                f"{producto['marca']} {producto['modelo']}"
            ),
            "cantidad": cantidad,
            "precio_unitario_bs": float(precio),
            "subtotal_bs": float(importe)
        })

    subtotal = dinero(subtotal)
    descuento_bs = dinero(
        subtotal * descuento / Decimal("100")
    )
    total = dinero(subtotal - descuento_bs)

    return {
        "estado": "cotizacion_generada",
        "productos": detalle,
        "subtotal_bs": float(subtotal),
        "descuento_pct": float(descuento),
        "descuento_bs": float(descuento_bs),
        "total_bs": float(total),
        "moneda": "BOB",
        "fuente": "Catálogo ficticio de demostración",
        "nota": (
            "Cotización informativa. "
            "No reserva productos ni modifica el stock."
        )
    }
