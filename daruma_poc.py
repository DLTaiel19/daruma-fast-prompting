# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # De la idea a la cocina — POC de Fast Prompting
# **Daruma Deco · Muebles a medida** — Segunda entrega · *Fast Prompting en Acción*
#
# Autor: **Lautaro Taiel Domínguez** · Comisión **96165** · Inteligencia Artificial: Generación de Prompts
#
# Esta notebook demuestra cómo, con **una sola llamada a la API** y técnicas de *fast prompting*,
# transformamos el mensaje suelto de un cliente en un **pack de venta completo**: un brief ordenado,
# el prompt para generar la imagen (texto→imagen) y los textos para WhatsApp e Instagram.
#
# > La documentación completa (problema, objetivos, metodología y viabilidad) está en el `README.md`.

# %% [markdown]
# ## 0. Configuración
# Dos interruptores controlan el **modo de ejecución** y el **costo**:
#
# - `MODO_DEMO = True` → usa una respuesta de ejemplo: **0 llamadas** a la API, no necesita API key. Ideal para revisar la notebook.
# - `MODO_DEMO = False` → llama a la API real (requiere la variable de entorno `OPENAI_API_KEY`).
# - `GENERAR_IMAGEN_API = False` → imprime el prompt listo para pegar en **Nightcafe** (gratis, 0 llamadas).
# - `GENERAR_IMAGEN_API = True` → genera la imagen con **GPT Image** (tiene costo).

# %%
import os
import json
import textwrap

# --- Configuración general ---
MODELO_TEXTO = "gpt-4o-mini"        # barato y más que suficiente para esta tarea
MODELO_IMAGEN = "gpt-image-1-mini"  # DALL·E 3 se retiró de la API el 12/05/2026; GPT Image es el reemplazo actual

MODO_DEMO = True             # True = respuesta de ejemplo (0 llamadas). False = API real.
GENERAR_IMAGEN_API = False   # False = prompt para Nightcafe (gratis). True = imagen con GPT Image (costo).
EJECUTAR_INTERACTIVO = False # True = habilita la carga interactiva del prompt (widgets / input()).

# La API key NUNCA se escribe en el código: se lee de una variable de entorno.
API_KEY = os.getenv("OPENAI_API_KEY", "")

# Contador global de llamadas, para el análisis de costos.
LLAMADAS_API = 0

print("Configuración cargada.")
print(f"  Modelo de texto : {MODELO_TEXTO}")
print(f"  Modelo de imagen: {MODELO_IMAGEN}")
print(f"  MODO_DEMO       : {MODO_DEMO}")

# %% [markdown]
# ## 1. Técnicas de *fast prompting* aplicadas
#
# 1. **Rol / persona** — el modelo actúa como asistente de Daruma Deco: responde en el tono y el contexto correctos.
# 2. **Delimitadores** (`"""`) — separan el mensaje del cliente de las instrucciones (claridad y seguridad ante inyección de texto).
# 3. **Salida estructurada (JSON)** — pedimos un JSON con claves fijas; se parsea directo y habilita la automatización.
# 4. **Few-shot (1 ejemplo)** — un ejemplo corto fija formato y estilo, y reduce los reintentos.
# 5. **Restricciones anti-alucinación** — "no inventes medidas; lo que falte va en `falta_confirmar`".
# 6. **Prompt único (la clave del costo)** — un solo prompt devuelve las 4 etapas juntas → **1 llamada** en vez de 4.
# 7. **Parámetros** — `response_format=json_object`, `max_tokens` acotado y `temperature` moderada para controlar costo y consistencia.

# %%
# System prompt: rol + reglas + esquema de salida (una sola llamada devuelve TODO el pack).
SYSTEM_PROMPT = """Sos el asistente de diseño y ventas de Daruma Deco, un taller de muebles a medida
(cocinas, baños, vestidores, livings). Sello de la marca: hecho a mano, materia prima premium,
LED incluido. Tono cálido, cercano y prolijo, en español rioplatense.

A partir del MENSAJE DEL CLIENTE (delimitado por triple comilla) devolvés TODO el pack de venta
en UNA sola respuesta, en formato JSON válido y sin ningún texto adicional, con exactamente estas claves:

{
  "brief": {
    "para_imagen": [rasgos visuales: estilo, distribución general, colores, materiales, iluminación],
    "datos_tecnicos": [datos duros: tipo de espacio, medidas mencionadas, presupuesto],
    "falta_confirmar": [preguntas cortas para completar el brief]
  },
  "prompt_imagen": "prompt fotorrealista EN INGLÉS, una sola línea, para un modelo texto->imagen; incluí estilo, materiales, colores, 'warm LED', atmósfera y encuadre; NO incluyas medidas ni números",
  "negative_prompt": "términos a evitar en la imagen (ej: text, watermark, people, clutter, distorted proportions)",
  "mensaje_whatsapp": "3-4 líneas para el cliente presentando el concepto como referencia de estilo (no un render final) e invitando a ajustar detalles",
  "caption_instagram": "caption con gancho en la 1a línea, 3-4 líneas y 5-8 hashtags del rubro"
}

REGLAS:
- No inventes medidas, materiales ni datos que el cliente no mencionó: si faltan, van en "falta_confirmar".
- En "prompt_imagen" nunca pongas medidas exactas ni números del espacio (el modelo de imagen no los respeta).
- Devolvé SOLO el JSON, sin explicaciones ni comillas de bloque."""

# Few-shot: un ejemplo corto que fija formato y estilo.
FEWSHOT_USER = 'MENSAJE DEL CLIENTE:\n"""Quiero un placard para el dormitorio, algo sobrio, gris con puntas de madera. Pared de 2 metros más o menos."""'

FEWSHOT_ASSISTANT = json.dumps({
    "brief": {
        "para_imagen": ["estilo sobrio y moderno", "frentes gris mate con detalles en madera natural", "dormitorio cálido y ordenado"],
        "datos_tecnicos": ["placard de dormitorio", "pared de aprox. 2 m (a confirmar)"],
        "falta_confirmar": ["¿alto disponible?", "¿puertas corredizas o batientes?", "¿interior con cajonera?"]
    },
    "prompt_imagen": "Photorealistic modern bedroom wardrobe, matte grey fronts with natural wood accents, minimalist and cozy interior, warm LED lighting, soft daylight, high detail, wide angle",
    "negative_prompt": "text, watermark, people, clutter, distorted proportions",
    "mensaje_whatsapp": "¡Hola! Te preparé una referencia de estilo para tu placard: gris mate con detalles en madera, bien sobrio. Es una idea del clima, no el diseño final: ajustamos medidas y terminaciones juntos. ¿Te gusta por dónde va?",
    "caption_instagram": "Un placard que ordena y suma calidez ✨\nGris mate + madera natural, con LED incluido.\nHecho 100% a medida para tu espacio.\nEscribinos y lo diseñamos juntos.\n#mueblesamedida #placard #dormitorio #interiordesign #daruma #hechoamano #deco"
}, ensure_ascii=False)


def construir_mensajes(mensaje_cliente):
    """Arma la lista de mensajes: rol (system) + few-shot + pedido del cliente delimitado."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEWSHOT_USER},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT},
        {"role": "user", "content": f'MENSAJE DEL CLIENTE:\n"""{mensaje_cliente}"""'},
    ]

print("Prompt del sistema y few-shot listos.")

# %%
# Respuesta de ejemplo para MODO_DEMO (respeta el mismo esquema que pedimos al modelo).
RESPUESTA_DEMO = {
    "brief": {
        "para_imagen": [
            "estilo nórdico/escandinavo, cálido y luminoso",
            "distribución lineal (cocina alargada), isla opcional a confirmar",
            "paleta clara: blanco + madera clara",
            "frentes blancos mate y madera natural, mesada clara",
            "mucha luz natural y LED cálido"
        ],
        "datos_tecnicos": [
            "departamento, cocina alargada 'no muy grande' (medidas sin confirmar)",
            "presupuesto ajustado",
            "contexto: mudanza / renovación"
        ],
        "falta_confirmar": [
            "¿medidas del paño principal?",
            "¿hay ancho suficiente para una isla?",
            "¿electrodomésticos a integrar?",
            "¿tono de madera preferido (claro/medio)?"
        ]
    },
    "prompt_imagen": "Photorealistic interior of a small Scandinavian apartment kitchen, linear galley layout, white matte cabinetry with light natural oak wood fronts, light stone countertop, warm LED under-cabinet lighting, bright and airy atmosphere, soft natural daylight, minimalist and cozy, wide angle, high detail",
    "negative_prompt": "text, watermark, people, clutter, distorted proportions",
    "mensaje_whatsapp": "¡Hola! Gracias por escribirnos 🙌 Te armé una referencia de estilo para tu cocina: línea nórdica, blanco con madera clara y luz cálida. Es una idea del clima y los materiales, no el diseño final: con tus medidas la ajustamos a tu espacio. ¿Te gusta por dónde va?",
    "caption_instagram": "Cocinas chicas que se sienten enormes ✨\nLínea nórdica: blanco + madera clara y LED cálido incluido.\nCada cocina la hacemos 100% a medida para tu espacio.\n¿La imaginás en tu casa? Escribinos y la diseñamos juntos.\n#cocinasamedida #mueblesamedida #cocinanordica #deco #interiordesign #hechoamano #darumadeco"
}
print("Respuesta de ejemplo (demo) cargada.")

# %%
def generar_pack(mensaje_cliente):
    """Devuelve (pack:dict, usage). En MODO_DEMO no gasta ninguna llamada a la API."""
    global LLAMADAS_API

    if MODO_DEMO:
        return RESPUESTA_DEMO, None

    # Import diferido: solo se necesita openai cuando se usa la API real.
    from openai import OpenAI
    if not API_KEY:
        raise RuntimeError("Falta la variable de entorno OPENAI_API_KEY.")

    client = OpenAI(api_key=API_KEY)
    respuesta = client.chat.completions.create(
        model=MODELO_TEXTO,
        messages=construir_mensajes(mensaje_cliente),
        response_format={"type": "json_object"},  # fuerza JSON válido -> evita reintentos (ahorro)
        temperature=0.7,
        max_tokens=700,                            # techo de salida -> controla el costo
    )
    LLAMADAS_API += 1
    contenido = respuesta.choices[0].message.content
    return json.loads(contenido), respuesta.usage


def _envolver(texto, ancho=92):
    return "\n".join(textwrap.fill(l, ancho) for l in str(texto).split("\n"))


def mostrar_pack(pack):
    """Imprime el pack de venta de forma legible."""
    b = pack["brief"]
    print("=" * 92)
    print("BRIEF — PARA LA IMAGEN")
    for x in b["para_imagen"]:
        print("  •", x)
    print("\nBRIEF — DATOS TÉCNICOS (guían la fabricación, NO van a la imagen)")
    for x in b["datos_tecnicos"]:
        print("  •", x)
    print("\nBRIEF — FALTA CONFIRMAR")
    for x in b["falta_confirmar"]:
        print("  •", x)
    print("=" * 92)
    print("PROMPT DE IMAGEN (texto→imagen):")
    print(_envolver(pack["prompt_imagen"]))
    print("\nNEGATIVE PROMPT:")
    print(_envolver(pack["negative_prompt"]))
    print("=" * 92)
    print("MENSAJE DE WHATSAPP:")
    print(_envolver(pack["mensaje_whatsapp"]))
    print("\nCAPTION DE INSTAGRAM:")
    print(_envolver(pack["caption_instagram"]))
    print("=" * 92)

print("Funciones definidas.")

# %% [markdown]
# ### Corrida principal
# Un mensaje real y desordenado → todo el pack de venta en **una sola llamada**.

# %%
MENSAJE_EJEMPLO = (
    "Hola! Vi sus trabajos en IG. Me mudo y quiero rehacer la cocina. Es un depto, "
    "la cocina es tipo alargada, no muy grande. Me gustan los tonos claros, algo nórdico, "
    "madera clara con blanco. No sé si entra una isla. Quiero buena luz. "
    "Presupuesto ajustado pero que quede linda."
)

pack, uso = generar_pack(MENSAJE_EJEMPLO)
mostrar_pack(pack)
print(f"\nLlamadas a la API en esta corrida: {LLAMADAS_API}")

# %% [markdown]
# ## 2. La optimización: **1 llamada en vez de 4**
#
# En la Preentrega 1 el flujo eran 4 etapas encadenadas (intake → prompt de imagen → copy cliente → copy IG),
# lo que implicaba **hasta 4 llamadas** a la API por cliente. Con *fast prompting* las unificamos en **1 sola**.
# Abajo comparamos el costo estimado de ambos enfoques.

# %%
# Precios vigentes (USD por millón de tokens). Verificá siempre en openai.com/api/pricing.
PRECIOS = {"gpt-4o-mini": {"in": 0.15, "out": 0.60}}


def estimar_tokens(texto):
    """Heurística simple: ~4 caracteres por token."""
    return max(1, len(str(texto)) // 4)


def costo_texto(tok_in, tok_out, modelo="gpt-4o-mini"):
    p = PRECIOS[modelo]
    return tok_in / 1e6 * p["in"] + tok_out / 1e6 * p["out"]


# --- Piezas de salida (se reutilizan para estimar ambos enfoques) ---
brief_txt = json.dumps(RESPUESTA_DEMO["brief"], ensure_ascii=False)
img_txt = RESPUESTA_DEMO["prompt_imagen"] + " " + RESPUESTA_DEMO["negative_prompt"]
whats_txt = RESPUESTA_DEMO["mensaje_whatsapp"]
cap_txt = RESPUESTA_DEMO["caption_instagram"]

# --- Enfoque OPTIMIZADO (1 llamada) ---
opt_in = (estimar_tokens(SYSTEM_PROMPT) + estimar_tokens(FEWSHOT_USER)
          + estimar_tokens(FEWSHOT_ASSISTANT) + estimar_tokens(MENSAJE_EJEMPLO))
opt_out = estimar_tokens(json.dumps(RESPUESTA_DEMO, ensure_ascii=False))
opt_costo = costo_texto(opt_in, opt_out)

# --- Variante OPTIMIZADA SIN few-shot (una llamada, menos tokens, algo menos de fiabilidad) ---
optnf_in = estimar_tokens(SYSTEM_PROMPT) + estimar_tokens(MENSAJE_EJEMPLO)
optnf_out = opt_out
optnf_costo = costo_texto(optnf_in, optnf_out)

# --- Enfoque NAÏVE (4 llamadas encadenadas) ---
# La ineficiencia real del encadenado: (1) el rol/contexto de la marca se repite en cada etapa,
# y (2) cada etapa posterior debe REENVIAR como input las salidas de las etapas anteriores.
PREAMBULO_COMUN = ("Sos el asistente de diseño y ventas de Daruma Deco, muebles a medida. "
                   "Sello: hecho a mano, materia prima premium, LED incluido. Tono cálido y prolijo.")
INSTR_NAIVE = {
    "intake": "Ordená el pedido del cliente en un brief con datos visuales, datos técnicos y qué falta confirmar.",
    "prompt_imagen": "A partir del brief, escribí un prompt fotorrealista en inglés y un negative prompt (sin medidas).",
    "copy_cliente": "A partir del brief y el concepto, escribí un mensaje de WhatsApp de 3-4 líneas.",
    "copy_ig": "A partir del brief y el concepto, escribí un caption de Instagram con 5-8 hashtags.",
}
pre = estimar_tokens(PREAMBULO_COMUN)  # se repite en las 4 etapas

# input por etapa (el rol se repite; el brief y el concepto se reenvían aguas abajo)
in_s1 = pre + estimar_tokens(INSTR_NAIVE["intake"]) + estimar_tokens(MENSAJE_EJEMPLO)
in_s2 = pre + estimar_tokens(INSTR_NAIVE["prompt_imagen"]) + estimar_tokens(brief_txt)
in_s3 = pre + estimar_tokens(INSTR_NAIVE["copy_cliente"]) + estimar_tokens(brief_txt) + estimar_tokens(img_txt)
in_s4 = pre + estimar_tokens(INSTR_NAIVE["copy_ig"]) + estimar_tokens(brief_txt) + estimar_tokens(img_txt)
naive_in = in_s1 + in_s2 + in_s3 + in_s4

# output por etapa
naive_out = (estimar_tokens(brief_txt) + estimar_tokens(img_txt)
             + estimar_tokens(whats_txt) + estimar_tokens(cap_txt))
naive_costo = costo_texto(naive_in, naive_out)

print("COMPARACIÓN DE COSTO (estimado, 1 cliente)")
print("-" * 72)
print(f"{'Enfoque':<28}{'Llamadas':>9}{'Tok in':>9}{'Tok out':>9}{'USD':>14}")
print(f"{'Naïve (encadenado)':<28}{4:>9}{naive_in:>9}{naive_out:>9}{naive_costo:>14.6f}")
print(f"{'Optimizado + few-shot':<28}{1:>9}{opt_in:>9}{opt_out:>9}{opt_costo:>14.6f}")
print(f"{'Optimizado sin few-shot':<28}{1:>9}{optnf_in:>9}{optnf_out:>9}{optnf_costo:>14.6f}")
print("-" * 72)
print(f"Reducción de llamadas: 4 -> 1  (75% menos requests por cliente).")
print()
print("Lecturas del experimento:")
print("  • La ganancia más sólida es la CANTIDAD DE LLAMADAS: menos latencia, menos")
print("    puntos de falla y más margen frente a los límites de la API.")
print("  • En tokens, unificar ahorra el contexto repetido del encadenado, pero el")
print("    few-shot vuelve a sumar entrada: por eso el costo queda casi igual al naïve.")
print("  • El few-shot es una PALANCA: sin él, la versión de 1 llamada es la más barata")
print(f"    (USD {optnf_costo:.6f}), a cambio de algo menos de consistencia en el formato.")
print()
print("Proyección a 1.000 clientes (llamadas):")
print(f"  Naïve      -> {4000:>5} llamadas")
print(f"  Optimizado -> {1000:>5} llamadas   (3.000 requests menos)")

# %% [markdown]
# ## 3. Generación de la imagen (texto→imagen)
#
# El prompt de imagen ya viene resuelto dentro del pack (etapa de texto). Ahora hay dos caminos:
# - **Gratis (recomendado para probar):** copiar el prompt en **Nightcafe** → **0 llamadas** a la API.
# - **Automático (con costo):** generar con **GPT Image** desde la misma API.

# %%
def paso_imagen(pack):
    """Genera la imagen o imprime el prompt para Nightcafe, según GENERAR_IMAGEN_API."""
    global LLAMADAS_API
    prompt = pack["prompt_imagen"]
    negativo = pack["negative_prompt"]

    if not GENERAR_IMAGEN_API:
        print("MODO GRATIS — pegá esto en Nightcafe (u otra herramienta gratuita):\n")
        print("PROMPT:\n" + _envolver(prompt))
        print("\nNEGATIVE PROMPT:\n" + _envolver(negativo))
        print("\n(0 llamadas a la API)")
        return None

    # Camino con costo: GPT Image (reemplazo de DALL·E en la API).
    from openai import OpenAI
    if not API_KEY:
        raise RuntimeError("Falta la variable de entorno OPENAI_API_KEY.")
    client = OpenAI(api_key=API_KEY)
    resultado = client.images.generate(
        model=MODELO_IMAGEN,
        prompt=f"{prompt}. Avoid: {negativo}.",  # GPT Image no tiene campo negative: se integra al prompt
        size="1024x1024",
        n=1,
    )
    LLAMADAS_API += 1

    import base64
    dato = resultado.data[0]
    ruta = "salidas/concepto_cocina.png"
    if getattr(dato, "b64_json", None):
        with open(ruta, "wb") as f:
            f.write(base64.b64decode(dato.b64_json))
        print(f"Imagen guardada en: {ruta}")
    else:
        print("URL de la imagen:", getattr(dato, "url", "(sin url)"))
    return ruta


_ = paso_imagen(pack)

# %% [markdown]
# ## 4. (Extra) Carga interactiva del prompt
# Para que el prompt del cliente **no quede hardcodeado** en la celda, se puede cargar de forma interactiva.
# Se activa con `EJECUTAR_INTERACTIVO = True`. Usa `ipywidgets` si está disponible; si no, cae en `input()`.

# %%
if EJECUTAR_INTERACTIVO:
    try:
        import ipywidgets as widgets
        from IPython.display import display

        caja = widgets.Textarea(
            placeholder="Escribí acá el mensaje del cliente...",
            layout=widgets.Layout(width="100%", height="90px"),
        )
        boton = widgets.Button(description="Generar pack", button_style="warning")
        salida = widgets.Output()

        def _on_click(_):
            with salida:
                salida.clear_output()
                p, _u = generar_pack(caja.value or MENSAJE_EJEMPLO)
                mostrar_pack(p)

        boton.on_click(_on_click)
        display(caja, boton, salida)
    except ImportError:
        texto = input("Mensaje del cliente: ")
        p, _u = generar_pack(texto or MENSAJE_EJEMPLO)
        mostrar_pack(p)
else:
    print("Interactividad desactivada (EJECUTAR_INTERACTIVO = False).")

# %% [markdown]
# ## 5. Conclusión — ¿el *fast prompting* mejoró la propuesta de la Preentrega 1?
#
# **Sí, en varios frentes:**
#
# - **Costo / eficiencia:** pasamos de un flujo de hasta **4 llamadas** por cliente a **1 sola** llamada
#   que devuelve todo el pack. A escala, esto es lo que hace al proyecto rentable.
# - **Confiabilidad:** al pedir **JSON estructurado** (`response_format=json_object`) y dar un **few-shot**,
#   la salida es consistente y parseable, sin reintentos que gasten tokens.
# - **Seguridad y claridad:** los **delimitadores** aíslan el mensaje del cliente de las instrucciones.
# - **Fidelidad al problema:** mantenemos la decisión clave de la Preentrega 1 —separar lo visual de lo técnico—
#   y evitamos pedirle medidas al modelo de imagen (que no las respeta).
# - **Adaptación al contexto real:** como **DALL·E 3 se retiró de la API (12/05/2026)**, la solución usa
#   **GPT Image** o, sin costo, **Nightcafe**, sin cambiar el resto del flujo.
#
# **Próximos pasos:** afinar el few-shot con casos reales de Daruma, validar el realismo de las imágenes
# con el criterio de los dueños, y medir el costo real por consulta con la facturación de la API.
