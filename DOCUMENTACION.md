# Guanaco Advisor — Documentación Técnica

> Asistente de preguntas y respuestas (RAG) **100 % local** sobre documentos
> propios e información de Odoo, con modelos de IA corriendo en infraestructura
> propia (sin enviar datos a la nube).

**Versión del documento:** 1.0 · **Fecha:** 2026-06-26

---

## 1. Resumen ejecutivo

**Guanaco Advisor** es un sistema de **RAG** (*Retrieval-Augmented Generation*) que
permite hacer preguntas en lenguaje natural y obtener respuestas basadas en:

- **Documentos propios**: PDF, TXT y DOCX (incluyendo texto dentro de imágenes vía OCR).
- **Datos de Odoo**: registros del ERP (ej. contactos de `res.partner`).

La clave del proyecto es que **todo corre localmente**: los modelos de lenguaje
y de embeddings se ejecutan con **Ollama** en el propio servidor, y la base de
conocimiento vive en una base vectorial local (**Chroma**). No hay dependencia
de APIs externas ni se exponen datos sensibles a terceros.

**¿Qué problema resuelve?**
Convierte información dispersa (documentos, ERP) en un asistente conversacional
que responde con precisión y **cita las fuentes** que usó para responder.

---

## 2. ¿Qué es RAG y por qué lo usamos?

Un modelo de lenguaje (LLM) por sí solo no conoce tus documentos ni tu base de
datos. El patrón **RAG** resuelve esto en dos tiempos:

1. **Recuperación (Retrieval):** ante una pregunta, se buscan los fragmentos de
   información más relevantes dentro de una base vectorial.
2. **Generación (Generation):** esos fragmentos se le pasan al LLM como
   *contexto*, y el modelo redacta la respuesta basándose en ellos.

```
Pregunta ─► [Buscar fragmentos relevantes] ─► [LLM redacta con ese contexto] ─► Respuesta + Fuentes
```

**Ventajas frente a un LLM "a secas":**

| Sin RAG | Con RAG (este proyecto) |
|---------|-------------------------|
| Responde de memoria (puede "alucinar") | Responde con datos reales y verificables |
| No conoce tus documentos | Usa tus PDFs, DOCX y Odoo |
| No cita fuentes | Cita de qué documento/registro salió la respuesta |
| Reentrenar es caro | Agregar conocimiento = solo indexar más documentos |

---

## 3. Arquitectura general

```
                          ┌──────────────────────────────────────────┐
                          │                NAVEGADOR                   │
                          │   chat.html  (interfaz, identidad ADEN)    │
                          └──────────────────┬─────────────────────────┘
                                             │  HTTP (JSON)
                                             ▼
                          ┌──────────────────────────────────────────┐
                          │              api.py  (FastAPI)             │
                          │   GET /      → sirve el chat               │
                          │   GET /salud → estado del sistema          │
                          │   POST /preguntar → consulta RAG           │
                          └───────┬───────────────────────┬────────────┘
                                  │                       │
              busca contexto      │                       │  redacta respuesta
                                  ▼                       ▼
                   ┌────────────────────────┐   ┌────────────────────────┐
                   │   Chroma (vectorial)   │   │   Ollama (modelos IA)   │
                   │   ./chroma_db          │   │   localhost:11434       │
                   │   colección "mis_docs" │   │   LLM + Embeddings      │
                   └───────────▲────────────┘   └────────────────────────┘
                               │ indexa
            ┌──────────────────┴───────────────────┐
            │                                       │
   ┌────────────────────┐                ┌────────────────────────┐
   │   contexto.py      │                │     odoo_sync.py        │
   │  Documentos:       │                │  ERP Odoo 13 (XML-RPC)  │
   │  PDF / TXT / DOCX  │                │  modelo res.partner     │
   │  + OCR (Tesseract) │                │                         │
   └────────────────────┘                └────────────────────────┘
```

**Componentes:**

| Componente | Tecnología | Rol |
|------------|-----------|-----|
| **Interfaz** | HTML/CSS/JS (`chat.html`) | Chat web con identidad ADEN |
| **API** | FastAPI + Uvicorn (`api.py`) | Expone el RAG por HTTP |
| **Orquestación RAG** | LlamaIndex | Une búsqueda + LLM |
| **Base vectorial** | Chroma (persistente) | Almacena los embeddings |
| **Motor de IA** | Ollama | Corre LLM y embeddings localmente |
| **Ingesta de documentos** | `contexto.py` + Tesseract OCR | Lee y vectoriza archivos |
| **Ingesta de Odoo** | `odoo_sync.py` (XML-RPC) | Trae registros del ERP |

---

## 4. Modelos de IA utilizados

Todos los modelos se ejecutan mediante **Ollama** en `localhost:11434`.

| Función | Modelo | Para qué se usa |
|---------|--------|-----------------|
| **LLM (generación)** | `gemma4:e2b` | Redacta la respuesta final a partir del contexto |
| **Embeddings** | `nomic-embed-text` | Convierte texto en vectores para la búsqueda semántica |
| **Visión (opcional)** | `llava` | Describe imágenes embebidas en documentos (desactivado por defecto) |

**¿Qué es un embedding?**
Es la representación numérica (un vector de cientos de dimensiones) del
*significado* de un texto. Textos con significado parecido quedan "cerca" en
ese espacio vectorial, lo que permite buscar por similitud semántica en lugar
de por palabras exactas.

> **Nota de configuración:** los nombres de los modelos están centralizados en
> `contexto.py` y son la única fuente de verdad: tanto la API como la
> sincronización de Odoo los reutilizan desde ahí.

---

## 5. Flujo de funcionamiento

### 5.1. Indexación (carga de conocimiento) — *offline*

Ocurre cuando se ejecuta `contexto.py` (documentos) u `odoo_sync.py` (Odoo).

```
1. Leer fuente        → archivo (PDF/TXT/DOCX) o registro de Odoo
2. Extraer texto      → incluye OCR de imágenes si aplica
3. Trocear (chunking) → LlamaIndex divide el texto en fragmentos
4. Generar embeddings → nomic-embed-text vectoriza cada fragmento
5. Guardar en Chroma  → vector + texto + metadatos (fuente, origen)
```

Cada fragmento se guarda con **metadatos** que permiten la trazabilidad:
`fuente` (nombre de archivo o `odoo:res.partner:123`), `origen` (`documento`
u `odoo`), etc.

### 5.2. Consulta (uso diario) — *online*

Ocurre en cada pregunta del usuario a través de la API.

```
1. Usuario escribe la pregunta en el chat
2. POST /preguntar → la API recibe el texto
3. La pregunta se convierte en embedding (nomic-embed-text)
4. Chroma devuelve los TOP_K (=4) fragmentos más similares
5. Esos fragmentos + la pregunta se le pasan al LLM (gemma4:e2b)
6. El LLM redacta la respuesta
7. Se devuelve { respuesta, fuentes } al navegador
```

El parámetro **`TOP_K = 4`** define cuántos fragmentos de contexto se recuperan
por pregunta (balance entre precisión y costo de cómputo).

---

## 6. Documentación de la API

API HTTP construida con **FastAPI**. Por defecto escucha en
`http://0.0.0.0:8000`. El índice y los modelos se cargan **una sola vez** al
arrancar el servidor; cada request solo hace la búsqueda y la generación.

### `GET /`
Sirve la interfaz de chat (`chat.html`).

- **Respuesta:** `text/html`

### `GET /salud`
Chequeo de estado del sistema y de la base vectorial.

- **Respuesta:** `application/json`

```json
{
  "estado": "ok",
  "modelo_llm": "gemma4:e2b",
  "modelo_embed": "nomic-embed-text",
  "vectores": 111000
}
```

### `POST /preguntar`
Realiza una consulta RAG.

- **Body (request):** `application/json`

```json
{ "pregunta": "¿Qué contactos hay en Buenos Aires?" }
```

- **Respuesta:** `application/json`

```json
{
  "respuesta": "Según los registros, los contactos en Buenos Aires son...",
  "fuentes": [
    { "fuente": "odoo:res.partner:123", "score": 0.8421 },
    { "fuente": "manual_inscripcion.pdf", "score": 0.7733 }
  ]
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `respuesta` | string | Texto generado por el LLM |
| `fuentes` | array | Fragmentos usados, con su `score` de similitud (trazabilidad) |

> **Importante:** la API **no** reindexa. Para cargar conocimiento nuevo se
> corre `contexto.py` u `odoo_sync.py` y luego se **reinicia** la API.

### Ejemplo de consumo (cURL)

```bash
curl -X POST http://localhost:8000/preguntar \
     -H "Content-Type: application/json" \
     -d "{\"pregunta\": \"¿Cuántos contactos hay?\"}"
```

---

## 7. Integración con Odoo

Módulo `odoo_sync.py`. Conecta a un **Odoo 13 self-hosted** vía **XML-RPC**
(usando la librería estándar `xmlrpc.client`, sin dependencias extra).

**Enfoque elegido:** *indexación / sincronización*. Los registros de Odoo se
traen, se convierten a texto legible y se indexan en la misma base Chroma que
los documentos. Así el RAG responde sobre el ERP igual que sobre los PDFs.

**Modelo sincronizado:** `res.partner` (contactos).

**Flujo de sincronización:**

```
1. Autenticación XML-RPC con Odoo (credenciales en .env)
2. Por cada modelo configurado:
   a. Borra los vectores previos de ese modelo (re-sync limpio)
   b. Trae los registros paginados (search_read, tandas de 200)
   c. Convierte cada registro a texto legible
      (resuelve relaciones many2one a su nombre)
   d. Indexa en Chroma con metadatos de trazabilidad
```

**Modos de uso:**

```bash
py odoo_sync.py --campos res.partner       # descubrir campos disponibles
py odoo_sync.py --listar-modelos partner   # buscar modelos por palabra clave
py odoo_sync.py                            # sincronizar e indexar
```

**Seguridad:** las credenciales (URL, base, usuario, API key) viven en un
archivo `.env` que **no se versiona** (está en `.gitignore`). Se recomienda un
usuario Odoo dedicado de **solo lectura**.

---

## 8. Interfaz de usuario

Archivo `chat.html` — una SPA mínima (HTML + CSS + JS, sin frameworks ni
dependencias externas). Características:

- **Identidad visual ADEN**: logo institucional (SVG embebido) y paleta de
  marca (rojo `#B31D15`, grises `#535353` / `#838383`, blanco y negro).
- **Chat en tiempo real** con autoajuste del cuadro de texto.
- **Indicador de estado** (vectores cargados, modelo activo).
- **Citas de fuentes**: muestra de qué documento/registro salió cada respuesta.

---

## 9. Puesta en marcha (deployment local)

El arranque está automatizado en **`iniciar.bat`**, que en orden:

```
1. Inicia Ollama (si no está corriendo)
2. Precarga el modelo gemma4:e2b en GPU (queda fijo en memoria)
3. Programa abrir el navegador en http://localhost:8000
4. Activa el entorno virtual y levanta la API (api.py)
```

**Requisitos del entorno:**

- **Ollama** instalado con los modelos `gemma4:e2b` y `nomic-embed-text`.
- **Python 3** con entorno virtual (`venv`) y las dependencias del proyecto
  (LlamaIndex, Chroma, FastAPI, Uvicorn, python-dotenv, etc.).
- **Tesseract-OCR** (opcional, solo para OCR de imágenes en documentos).

---

## 10. Stack tecnológico (resumen)

| Capa | Tecnología |
|------|-----------|
| Lenguaje | Python 3 |
| Motor de IA local | Ollama |
| LLM | gemma4:e2b |
| Embeddings | nomic-embed-text |
| Orquestación RAG | LlamaIndex |
| Base vectorial | Chroma (persistente) |
| API web | FastAPI + Uvicorn |
| Frontend | HTML/CSS/JS (vanilla) |
| OCR | Tesseract |
| Integración ERP | Odoo 13 vía XML-RPC |
| Gestión de secretos | python-dotenv (`.env`) |

---

## 11. Características destacadas (para la presentación)

- 🔒 **100 % local y privado** — ningún dato sale de la infraestructura propia.
- 📚 **Multi-fuente** — combina documentos (PDF/DOCX/TXT) y datos del ERP Odoo.
- 🔍 **Trazabilidad** — cada respuesta cita las fuentes que la respaldan.
- 🖼️ **OCR integrado** — extrae texto incluso de imágenes dentro de documentos.
- ⚡ **Indexación incremental** — agregar conocimiento no requiere reentrenar.
- 🎨 **Interfaz con identidad ADEN** — lista para uso institucional.
- 🧩 **Modular y extensible** — sumar nuevas fuentes es agregar un script de ingesta.

---

## 12. Consideraciones y trabajo futuro

- **Escala de indexación:** la carga inicial de grandes volúmenes (ej.
  ~111.000 registros de Odoo) es intensiva en cómputo de embeddings. Se puede
  acelerar con *embeddings por lote* (batch) en lugar de uno por uno.
- **Sincronización incremental de Odoo:** hoy la re-sincronización reindexa el
  modelo completo; a futuro puede optimizarse usando la fecha de modificación
  (`write_date`) para traer solo lo que cambió.
- **Respuestas en streaming:** mostrar la respuesta del LLM token a token para
  mejorar la percepción de velocidad.
- **Control de acceso:** la API hoy es abierta en la red local; puede sumarse
  autenticación si se expone más allá.

---

*Documento generado como base para presentación del proyecto Guanaco Advisor.*
