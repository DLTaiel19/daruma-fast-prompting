# De la idea a la cocina — POC de *Fast Prompting* para Daruma Deco

**Segunda entrega · _Fast Prompting en Acción: Desentrañando la Magia_**

- **Autor:** Lautaro Taiel Domínguez
- **Comisión:** 96165
- **Materia:** Inteligencia Artificial: Generación de Prompts
- **Caso de estudio:** [Daruma Deco](https://instagram.com/darumadeco_) · muebles a medida (cocinas)

Prueba de concepto en Jupyter Notebook que aplica técnicas de *fast prompting* para transformar,
**con una sola llamada a la API**, el mensaje suelto de un cliente en un **pack de venta completo**:
un brief ordenado, el prompt para generar la imagen (texto→imagen) y los textos para WhatsApp e Instagram.

> El código está en [`daruma_poc.ipynb`](daruma_poc.ipynb). La notebook viene ejecutada en **modo demo**
> (respuesta de ejemplo, sin costo) para poder revisarla sin API key.

---

## 1. Introducción

### 1.1 Nombre del proyecto
**De la idea a la cocina** — asistente de conceptos visuales con IA para Daruma Deco.

### 1.2 Problema a abordar
Daruma Deco fabrica muebles **a medida**, por lo que su producto **no existe hasta que se fabrica**.
En el caso de las cocinas —el producto más destacado y el de mayor carga económica y emocional— hay
una **brecha grande entre lo que el cliente dice y lo que logra imaginar**. Un mensaje típico como
*"quiero algo moderno, blanco con madera, cálido, no sé si entra una isla"* no devuelve ninguna imagen
concreta. Esa incertidumbre frena la decisión, desalinea expectativas y obliga al emprendimiento a
invertir tiempo describiendo ideas o buscando fotos de referencia sueltas.

Es **relevante** resolverlo porque una referencia visual temprana alinea al cliente con el taller,
acelera la venta, reduce el retrabajo y, de paso, genera material para las redes que Daruma ya produce.

### 1.3 Propuesta de solución y vínculo con la IA
Un asistente basado en prompts que acompaña la conversación de venta y combina los **dos modelos del curso**:

- **Texto→texto** (OpenAI, `gpt-4o-mini`): ordena el pedido y produce el brief, el prompt de imagen y los copys.
- **Texto→imagen** (GPT Image o **Nightcafe**): genera el concepto visual de la cocina.

**Decisión de alcance clave:** la imagen es una **referencia de estilo y clima**, no un plano ni un render
de fabricación. Por eso las variables se separan en dos grupos: las **visuales** (estilo, materiales,
colores, iluminación) alimentan la imagen; las **técnicas** (medidas, restricciones) se registran como
dato estructurado para la fabricación, pero **no** se le piden al modelo de imagen (no las respeta).

### 1.4 Viabilidad
El proyecto es realizable con recursos accesibles y sin infraestructura compleja: se ejecuta en una
notebook, con la API de OpenAI para texto y una herramienta **gratuita** (Nightcafe) o **muy barata**
(GPT Image, ~USD 0,005/imagen) para las imágenes. Las cuentas nuevas de OpenAI incluyen **USD 5 de
crédito gratis**, suficiente para toda la POC. El alcance está acotado a un caso real (cocinas) y a un
flujo optimizado a **una sola llamada** por cliente.

> **Nota de contexto:** DALL·E 3 fue **retirado de la API de OpenAI el 12/05/2026**. Por eso la
> notebook usa **GPT Image** como reemplazo integrado, y deja **Nightcafe** como opción gratuita.

---

## 2. Objetivos

**General:** demostrar, con una POC funcional, cómo el *fast prompting* resuelve el problema de
pre-visualización de cocinas de forma eficaz y **rentable** (mínimas llamadas a la API).

**Específicos:**
1. Aplicar técnicas de *fast prompting* (rol, delimitadores, few-shot, salida estructurada, restricciones).
2. **Unificar** el flujo de 4 etapas de la Preentrega 1 en **una sola llamada**.
3. Experimentar con distintas configuraciones (con/sin few-shot) y medir su impacto en costo y fiabilidad.
4. Comparar el costo del enfoque encadenado vs. el optimizado.
5. Dejar la generación de imagen resuelta por dos vías: gratuita (Nightcafe) y por API (GPT Image).

---

## 3. Metodología

1. **Fraccionar el problema** en etapas (intake → prompt de imagen → copy) para entenderlo.
2. **Prototipar** cada prompt en ChatGPT antes de llevarlo a código.
3. **Optimizar (fast prompting):** unificar las etapas en **un único prompt** que devuelve un **JSON**
   con todo el pack, reduciendo de 4 llamadas a 1.
4. **Instrumentar el costo:** contar llamadas y estimar tokens para comparar enfoques (celda de análisis).
5. **Modo demo:** un interruptor (`MODO_DEMO`) permite revisar la notebook con una respuesta de ejemplo,
   sin gastar llamadas ni requerir API key; al desactivarlo, se conecta a la API real.
6. **Validación con el caso real:** el criterio de los dueños de Daruma valida si los conceptos son realistas.

---

## 4. Herramientas y tecnologías

**Stack:** Python · Jupyter Notebook · API de OpenAI (`gpt-4o-mini`) · GPT Image / Nightcafe.

**Técnicas de *fast prompting* utilizadas (y por qué):**

| Técnica | Para qué sirve en este proyecto |
|---|---|
| **Rol / persona** | El modelo responde como asistente de Daruma: tono y contexto correctos. |
| **Delimitadores (`"""`)** | Aíslan el mensaje del cliente de las instrucciones: claridad y seguridad ante inyección. |
| **Salida estructurada (JSON)** | Claves fijas → se parsea directo y habilita la automatización. |
| **Few-shot (1 ejemplo)** | Fija formato y estilo, reduce reintentos. Es una **palanca** de costo/fiabilidad. |
| **Restricciones anti-alucinación** | "No inventes medidas; lo que falte va en `falta_confirmar`". |
| **Prompt único** | Devuelve las 4 etapas juntas → **1 llamada** en lugar de 4 (clave de la rentabilidad). |
| **Parámetros** | `response_format=json_object` (evita reintentos), `max_tokens` acotado y `temperature` moderada. |

**¿Por qué `gpt-4o-mini`?** Es el modelo de texto más económico de OpenAI (USD 0,15/0,60 por millón de
tokens de entrada/salida) y sobra para una tarea de estructuración y redacción como esta.

---

## 5. Implementación

Toda la implementación (prompts + código) está en **[`daruma_poc.ipynb`](daruma_poc.ipynb)**, organizada así:

- **0. Configuración** — interruptores de modo y costo.
- **1. Técnicas + prompt del sistema + few-shot** — el corazón del *fast prompting*.
- **Corrida principal** — un mensaje real → pack completo en 1 llamada.
- **2. Análisis de costo** — naïve (4 llamadas) vs optimizado (1), con y sin few-shot.
- **3. Imagen** — prompt para Nightcafe (gratis) o generación con GPT Image (con costo).
- **4. (Extra) Interactividad** — carga del prompt por widget/`input()`, no hardcodeado.
- **5. Conclusión** — análisis de la mejora respecto de la Preentrega 1.

### Análisis de costos (resumen)

| Enfoque | Llamadas | USD / cliente (estimado) |
|---|---|---|
| Naïve (encadenado) | 4 | ~0,00036 |
| Optimizado + few-shot | **1** | ~0,00036 |
| Optimizado sin few-shot | **1** | ~0,00031 |

La ganancia más sólida es la **reducción de llamadas (4 → 1, 75% menos)**: menos latencia, menos puntos
de falla y más margen frente a los límites de la API. En tokens, unificar ahorra el contexto repetido del
encadenado; el few-shot vuelve a sumar entrada, por eso se lo trata como una palanca ajustable.

---

## 6. Cómo ejecutar

```bash
# 1) Clonar el repo
git clone <URL-de-tu-repo>
cd daruma-fast-prompting

# 2) (Opcional) entorno virtual
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Dependencias
pip install -r requirements.txt

# 4) Abrir la notebook
jupyter notebook daruma_poc.ipynb
```

**Modo demo (sin costo):** dejá `MODO_DEMO = True`. La notebook corre con una respuesta de ejemplo.

**Modo real (API):**
1. Conseguí una API key en <https://platform.openai.com>.
2. Cargala como variable de entorno (nunca la escribas en el código):
   ```bash
   export OPENAI_API_KEY="sk-..."      # Windows PowerShell: setx OPENAI_API_KEY "sk-..."
   ```
3. Poné `MODO_DEMO = False`. Para generar la imagen por API, `GENERAR_IMAGEN_API = True`
   (o dejalo en `False` y pegá el prompt en Nightcafe, gratis).

---

## 7. Estructura del repositorio

```
daruma-fast-prompting/
├── README.md
├── daruma_poc.ipynb      # POC (notebook, ejecutada en modo demo)
├── daruma_poc.py         # misma notebook en formato script (jupytext)
├── requirements.txt
├── .env.example
├── .gitignore
└── salidas/              # imágenes generadas (si se usa GPT Image)
```

## Cómo publicar en GitHub

```bash
cd daruma-fast-prompting
git init
git add .
git commit -m "Preentrega 2 - POC Fast Prompting (Daruma Deco)"
git branch -M main
git remote add origin https://github.com/<tu-usuario>/daruma-fast-prompting.git
git push -u origin main
```

Verificá que el repositorio quede **público** y entregá ese enlace.
