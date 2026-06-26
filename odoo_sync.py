#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sincronización de Odoo -> RAG (Chroma).

Se conecta a un Odoo self-hosted por XML-RPC (sin instalar nada: usa la
librería estándar 'xmlrpc.client'), trae los registros de los modelos que
configures (estudiantes, cursos, etc.), los convierte en texto y los indexa
en la MISMA base vectorial Chroma que usa contexto.py / api.py.

Así el RAG puede responder preguntas sobre la info de Odoo igual que lo hace
sobre tus PDFs y documentos.

USO:
    venv\\Scripts\\activate

    # 1) (Opcional) Ver los campos de res.partner para elegir cuáles indexar:
    py odoo_sync.py --campos res.partner

    # 2) (Opcional) Buscar otros modelos por palabra clave:
    py odoo_sync.py --listar-modelos partner

    # 3) Sincronizar (indexar). Por defecto es REANUDABLE: saltea los registros
    #    que ya estén en Chroma, así podés cortar (Ctrl+C) y retomar sin perder
    #    lo ya hecho ni re-generar esos embeddings.
    py odoo_sync.py

    # 4) Forzar re-sincronización COMPLETA (borra los vectores del modelo y
    #    vuelve a traer TODO). Útil para reflejar bajas/cambios masivos.
    py odoo_sync.py --reset

Después de sincronizar, reiniciá api.py para que tome los vectores nuevos.

RENDIMIENTO: los embeddings se generan por LOTE (batch) — se mandan EMBED_BATCH
        textos en cada llamada a Ollama, en vez de uno por uno. Esto acelera la
        carga inicial de muchos registros en un orden de magnitud.

CONFIG: las credenciales salen de un archivo .env (ver .env.example).
        Los modelos a indexar se definen abajo en MODELOS_ODOO.
"""

import os
import sys
import time
import logging
import xmlrpc.client

from dotenv import load_dotenv

# Reutilizamos la config de modelos y rutas del proyecto (única fuente de verdad)
from contexto import CARPETA_DB, COLECCION, MODELO_EMBED

# ----------------------------------------------------------------------------
# QUÉ MODELOS DE ODOO INDEXAR
# ----------------------------------------------------------------------------
# Cada entrada describe un modelo de Odoo a traer. Para ver/ajustar los campos:
#     py odoo_sync.py --campos res.partner
#
#   model   -> nombre técnico del modelo en Odoo (ej: "res.partner")
#   dominio -> filtro de registros estilo Odoo. [] = todos.
#              Ej. sólo personas (no empresas): [["is_company","=",False]]
#   campos  -> lista de campos a traer. [] = todos los legibles (más lento,
#              pero útil al principio para no perderte nada).
#
# Probado contra Odoo 13 (XML-RPC). Los campos de abajo existen en res.partner
# estándar; si tu base tiene campos a medida, agregalos a la lista.
MODELOS_ODOO = [
    {
        "model": "res.partner",
        "dominio": [],
        "campos": [
            "name", "display_name", "email", "phone", "mobile",
            "function", "title", "comment",
            "is_company", "company_type", "parent_id", "category_id",
            "street", "street2", "city", "state_id", "zip", "country_id",
            "vat", "website",
        ],
    },
]

# Cuántos registros traer por tanda de Odoo (paginado para no saturar red/memoria)
TANDA = 200

# Cuántos textos se mandan juntos en CADA llamada de embedding a Ollama.
# Más alto = menos llamadas HTTP = más rápido (a costa de algo más de RAM/VRAM).
EMBED_BATCH = 64

# Cada cuántos registros se confirma (inserta) en Chroma. Al confirmar seguido,
# si el proceso se corta, lo ya insertado queda guardado y se puede reanudar.
LOTE_INSERT = 500

# ----------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("odoo-sync")


def paso(msg):
    log.info("=" * 60)
    log.info(msg)
    log.info("=" * 60)


# ----------------------------------------------------------------------------
# CONEXIÓN A ODOO (XML-RPC)
# ----------------------------------------------------------------------------
def conectar():
    """Autentica contra Odoo y devuelve (url, db, uid, password, models_proxy)."""
    load_dotenv()
    url = os.getenv("ODOO_URL", "").rstrip("/")
    db = os.getenv("ODOO_DB", "")
    user = os.getenv("ODOO_USER", "")
    password = os.getenv("ODOO_API_KEY", "")

    faltan = [k for k, v in {
        "ODOO_URL": url, "ODOO_DB": db,
        "ODOO_USER": user, "ODOO_API_KEY": password,
    }.items() if not v]
    if faltan:
        log.error("Faltan variables en .env: %s", ", ".join(faltan))
        log.error("Copiá .env.example a .env y completalo.")
        sys.exit(1)

    log.info("Conectando a %s (db=%s, user=%s)...", url, db, user)
    common = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/common", allow_none=True)
    try:
        info = common.version()
        log.info("Odoo %s", info.get("server_version", "?"))
        uid = common.authenticate(db, user, password, {})
    except Exception as e:
        log.error("No se pudo conectar/autenticar: %s", e)
        sys.exit(1)

    if not uid:
        log.error("Autenticación fallida: revisá ODOO_USER / ODOO_API_KEY / ODOO_DB.")
        sys.exit(1)

    log.info("Autenticado OK (uid=%s)", uid)
    models = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/object", allow_none=True)
    return url, db, uid, password, models


def _ejecutar(models, db, uid, password, modelo, metodo, args, kwargs=None):
    """Atajo para models.execute_kw con manejo de errores."""
    return models.execute_kw(db, uid, password, modelo, metodo, args, kwargs or {})


# ----------------------------------------------------------------------------
# MODO DESCUBRIMIENTO (para configurar MODELOS_ODOO)
# ----------------------------------------------------------------------------
def listar_modelos(filtro=""):
    """Lista los modelos de Odoo cuyo nombre técnico contiene 'filtro'."""
    url, db, uid, password, models = conectar()
    dominio = [["model", "ilike", filtro]] if filtro else []
    res = _ejecutar(models, db, uid, password, "ir.model", "search_read",
                    [dominio], {"fields": ["model", "name"], "order": "model"})
    paso("Modelos encontrados: %d" % len(res))
    for r in res:
        print(f"  {r['model']:<35} {r['name']}")


def listar_campos(modelo):
    """Lista los campos de un modelo (nombre técnico, etiqueta y tipo)."""
    url, db, uid, password, models = conectar()
    campos = _ejecutar(models, db, uid, password, modelo, "fields_get",
                       [], {"attributes": ["string", "type"]})
    paso("Campos de '%s': %d" % (modelo, len(campos)))
    for nombre in sorted(campos):
        c = campos[nombre]
        print(f"  {nombre:<30} {c.get('type',''):<12} {c.get('string','')}")


# ----------------------------------------------------------------------------
# TRAER REGISTROS Y CONVERTIRLOS EN TEXTO
# ----------------------------------------------------------------------------
def _valor_legible(v):
    """Convierte un valor de Odoo a texto plano legible."""
    if v is None or v is False:
        return ""
    # many2one viene como [id, "Nombre"] -> nos quedamos con el nombre
    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], int) \
            and isinstance(v[1], str):
        return v[1]
    if isinstance(v, (list, tuple)):
        return ", ".join(str(_valor_legible(x)) for x in v if x not in (None, False))
    return str(v)


def registro_a_texto(modelo, registro, etiquetas):
    """Arma un bloque de texto legible a partir de un registro de Odoo."""
    rid = registro.get("id")
    titulo = registro.get("name") or registro.get("display_name") or f"{modelo} #{rid}"
    lineas = [f"[Odoo · {modelo} · ID {rid}] {titulo}"]
    for campo, valor in registro.items():
        if campo == "id":
            continue
        txt = _valor_legible(valor)
        if not txt:
            continue
        etiqueta = etiquetas.get(campo, campo)
        lineas.append(f"{etiqueta}: {txt}")
    return "\n".join(lineas)


def etiquetas_de(modelo, models, db, uid, password):
    """Devuelve {campo: etiqueta legible} del modelo (una sola consulta)."""
    try:
        meta = _ejecutar(models, db, uid, password, modelo, "fields_get",
                         [], {"attributes": ["string"]})
        return {k: v.get("string", k) for k, v in meta.items()}
    except Exception as e:
        log.warning("  [%s] no pude leer etiquetas de campos: %s", modelo, e)
        return {}


def contar_registros(spec, models, db, uid, password):
    """Cuántos registros matchea el dominio del modelo."""
    return _ejecutar(models, db, uid, password, spec["model"], "search_count",
                     [spec.get("dominio", [])])


def iter_registros(spec, models, db, uid, password):
    """Itera los registros del modelo en tandas (paginado), uno por uno.

    Ordena por 'id' para que el paginado sea estable (clave para reanudar sin
    saltear ni duplicar registros entre tandas).
    """
    modelo = spec["model"]
    dominio = spec.get("dominio", [])
    campos = spec.get("campos", [])
    offset = 0
    while True:
        kwargs = {"offset": offset, "limit": TANDA, "order": "id"}
        if campos:
            kwargs["fields"] = campos
        tanda = _ejecutar(models, db, uid, password, modelo, "search_read",
                          [dominio], kwargs)
        if not tanda:
            break
        for reg in tanda:
            yield reg
        offset += TANDA


# ----------------------------------------------------------------------------
# SINCRONIZACIÓN -> CHROMA  (por lotes + reanudable)
# ----------------------------------------------------------------------------
def ids_existentes(collection, modelo):
    """IDs de Odoo de este modelo que YA están indexados en Chroma."""
    got = collection.get(where={"modelo": modelo}, include=["metadatas"])
    return {m.get("odoo_id") for m in (got.get("metadatas") or []) if m}


def sincronizar(reset=False):
    """Indexa los MODELOS_ODOO en Chroma. Por lotes y reanudable.

    reset=False (defecto): saltea los registros ya presentes en Chroma
                           (REANUDABLE: cortar y volver a correr no re-trabaja).
    reset=True           : borra los vectores del modelo y vuelve a traer TODO.
    """
    # Imports pesados sólo cuando hace falta (igual que contexto.py)
    paso("Cargando librerías de indexado...")
    import chromadb
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.schema import TextNode
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.ollama import OllamaEmbedding

    # embed_batch_size => varios textos en una sola llamada HTTP a Ollama
    Settings.embed_model = OllamaEmbedding(
        model_name=MODELO_EMBED, embed_batch_size=EMBED_BATCH)

    url, db, uid, password, models = conectar()

    paso("Abriendo base vectorial Chroma")
    chroma = chromadb.PersistentClient(path=CARPETA_DB)
    collection = chroma.get_or_create_collection(COLECCION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    log.info("Vectores actuales en la colección: %d", collection.count())

    for spec in MODELOS_ODOO:
        modelo = spec["model"]
        paso("Sincronizando modelo: %s  [%s]"
             % (modelo, "reset" if reset else "reanudable"))
        t = time.time()

        if reset:
            try:
                collection.delete(where={"modelo": modelo})
                log.info("  [%s] vectores previos eliminados (reset)", modelo)
            except Exception as e:
                log.warning("  [%s] no pude limpiar vectores previos: %s", modelo, e)
            ya = set()
        else:
            ya = ids_existentes(collection, modelo)
            log.info("  [%s] ya indexados en Chroma: %d (se saltean)", modelo, len(ya))

        etiquetas = etiquetas_de(modelo, models, db, uid, password)
        try:
            total = contar_registros(spec, models, db, uid, password)
        except Exception as e:
            log.error("  [%s] ERROR contando registros: %s", modelo, e)
            continue
        log.info("  [%s] %d registro(s) en Odoo; faltarían ~%d",
                 modelo, total, max(0, total - len(ya)))

        buffer, nuevos, vistos = [], 0, 0
        try:
            for reg in iter_registros(spec, models, db, uid, password):
                vistos += 1
                if reg["id"] in ya:          # ya indexado -> reanudar, no repetir
                    continue
                texto = registro_a_texto(modelo, reg, etiquetas)
                buffer.append(TextNode(
                    text=texto,
                    metadata={
                        "fuente": f"odoo:{modelo}:{reg['id']}",
                        "origen": "odoo",
                        "modelo": modelo,
                        "odoo_id": reg["id"],
                    },
                ))
                nuevos += 1
                if len(buffer) >= LOTE_INSERT:
                    index.insert_nodes(buffer)   # embeddea el lote y lo guarda
                    buffer = []
                    log.info("  [%s] insertados %d nuevos (revisados %d/%d) en %.1fs",
                             modelo, nuevos, vistos, total, time.time() - t)
            if buffer:
                index.insert_nodes(buffer)
        except KeyboardInterrupt:
            if buffer:
                index.insert_nodes(buffer)       # guarda lo que haya en el buffer
            log.warning("  [%s] interrumpido: %d nuevos guardados. "
                        "Volvé a correr para reanudar desde acá.", modelo, nuevos)
            raise
        except Exception as e:
            log.error("  [%s] ERROR: %s  (lo ya insertado quedó guardado; "
                      "es reanudable: volvé a correr)", modelo, e)
            continue

        log.info("  [%s] listo: %d registro(s) nuevos indexados en %.1fs",
                 modelo, nuevos, time.time() - t)

    paso("Sincronización completa")
    log.info("Vectores en la colección ahora: %d", collection.count())
    log.info("Reiniciá api.py para que la API tome los datos nuevos.")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    if args and args[0] == "--listar-modelos":
        listar_modelos(args[1] if len(args) > 1 else "")
    elif args and args[0] == "--campos":
        if len(args) < 2:
            log.error("Uso: py odoo_sync.py --campos <modelo>  (ej: res.partner)")
            sys.exit(1)
        listar_campos(args[1])
    elif args and args[0] == "--reset":
        sincronizar(reset=True)
    else:
        sincronizar()


if __name__ == "__main__":
    main()
