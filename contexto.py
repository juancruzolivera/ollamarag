#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG local con Ollama + Chroma + LlamaIndex.
Lee PDFs, TXT y DOCX (incluyendo OCR de imágenes embebidas) de una carpeta,
los vectoriza en una base Chroma persistente y permite consultarlos.

Va tirando logs por consola en cada paso para ver el avance.
"""

import os
import io
import sys
import time
import zipfile
import logging

# ----------------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------------
CARPETA_DOCS   = "./mis_documentos"
CARPETA_DB     = "./chroma_db"
COLECCION      = "mis_docs"
MODELO_LLM     = "gemma4:e2b"
MODELO_EMBED   = "nomic-embed-text"
MODELO_VISION  = "llava"          # solo se usa si USAR_VISION = True
USAR_VISION    = False            # True = describir imágenes (lento en CPU)
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
IDIOMA_OCR     = "spa"
TOP_K          = 4

# ----------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("rag")


def paso(msg):
    """Marca visual para separar etapas grandes."""
    log.info("=" * 60)
    log.info(msg)
    log.info("=" * 60)


# ----------------------------------------------------------------------------
# IMPORTS PESADOS (con log para ver si tardan en cargar)
# ----------------------------------------------------------------------------
paso("Cargando librerías...")
t0 = time.time()

import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

log.info("Librerías cargadas en %.1fs", time.time() - t0)


# ----------------------------------------------------------------------------
# EXTRACCIÓN DE DOCX CON IMÁGENES
# ----------------------------------------------------------------------------
def describir_imagen_ollama(img_bytes, model=MODELO_VISION):
    import base64
    import requests
    b64 = base64.b64encode(img_bytes).decode()
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": "Describí en español qué muestra esta imagen, "
                      "incluyendo cualquier texto, dato o gráfico.",
            "images": [b64],
            "stream": False,
        },
        timeout=300,
    )
    return r.json().get("response", "")


def extraer_docx(ruta, usar_vision=USAR_VISION):
    """Texto del docx + OCR de cada imagen (+ descripción opcional)."""
    import docx2txt
    import pytesseract
    from PIL import Image

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    nombre = os.path.basename(ruta)
    partes = []

    # 1) Texto normal
    log.info("  [%s] extrayendo texto...", nombre)
    texto = docx2txt.process(ruta) or ""
    partes.append(texto)
    log.info("  [%s] texto extraído (%d caracteres)", nombre, len(texto))

    # 2) Imágenes embebidas (un .docx es un .zip por dentro)
    with zipfile.ZipFile(ruta) as z:
        imgs = [n for n in z.namelist() if n.startswith("word/media/")]
        log.info("  [%s] %d imagen(es) embebida(s)", nombre, len(imgs))

        for i, img_nombre in enumerate(imgs, 1):
            data = z.read(img_nombre)
            try:
                img = Image.open(io.BytesIO(data))
            except Exception as e:
                log.warning("  [%s] imagen %d/%d no se pudo abrir: %s",
                            nombre, i, len(imgs), e)
                continue

            # OCR
            try:
                t = time.time()
                ocr = pytesseract.image_to_string(img, lang=IDIOMA_OCR).strip()
                if ocr:
                    partes.append(f"\n[Texto en imagen {img_nombre}]:\n{ocr}")
                    log.info("  [%s] imagen %d/%d: OCR %d chars (%.1fs)",
                             nombre, i, len(imgs), len(ocr), time.time() - t)
                else:
                    log.info("  [%s] imagen %d/%d: OCR sin texto (%.1fs)",
                             nombre, i, len(imgs), time.time() - t)
            except Exception as e:
                log.warning("  [%s] imagen %d/%d: OCR falló: %s",
                            nombre, i, len(imgs), e)

            # Visión opcional
            if usar_vision:
                try:
                    t = time.time()
                    log.info("  [%s] imagen %d/%d: describiendo con %s...",
                             nombre, i, len(imgs), MODELO_VISION)
                    desc = describir_imagen_ollama(data)
                    if desc:
                        partes.append(f"\n[Descripción de imagen {img_nombre}]:\n{desc}")
                    log.info("  [%s] imagen %d/%d: descripción lista (%.1fs)",
                             nombre, i, len(imgs), time.time() - t)
                except Exception as e:
                    log.warning("  [%s] imagen %d/%d: visión falló: %s",
                                nombre, i, len(imgs), e)

    return "\n".join(partes)


EXTENSIONES = (".pdf", ".txt", ".docx")


def listar_archivos(carpeta):
    """Lista los archivos soportados de la carpeta (ordenados)."""
    if not os.path.isdir(carpeta):
        log.error("No existe la carpeta '%s'. Creala y poné tus archivos.", carpeta)
        sys.exit(1)
    return sorted(f for f in os.listdir(carpeta)
                  if f.lower().endswith(EXTENSIONES))


def cargar_documentos(carpeta, archivos):
    """Devuelve lista de Document de LlamaIndex (solo de 'archivos') con log."""
    docs = []
    for n, archivo in enumerate(archivos, 1):
        ruta = os.path.join(carpeta, archivo)
        log.info("--- Procesando %d/%d: %s ---", n, len(archivos), archivo)
        t = time.time()

        ext = archivo.lower().rsplit(".", 1)[-1]
        try:
            if ext == "docx":
                texto = extraer_docx(ruta)
            elif ext == "txt":
                with open(ruta, encoding="utf-8", errors="ignore") as f:
                    texto = f.read()
            elif ext == "pdf":
                from pypdf import PdfReader
                reader = PdfReader(ruta)
                log.info("  [%s] %d página(s)", archivo, len(reader.pages))
                texto = "\n".join((p.extract_text() or "") for p in reader.pages)
                if not texto.strip():
                    log.warning("  [%s] PDF sin texto extraíble "
                                "(¿escaneado? necesitaría OCR de página completa)",
                                archivo)
            else:
                continue

            docs.append(Document(text=texto, metadata={"fuente": archivo}))
            log.info("  [%s] listo: %d caracteres en %.1fs",
                     archivo, len(texto), time.time() - t)
        except Exception as e:
            log.error("  [%s] ERROR al procesar: %s", archivo, e)

    log.info("Documentos cargados: %d", len(docs))
    return docs


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    paso("Configurando modelos Ollama")
    log.info("LLM=%s  Embeddings=%s  Visión=%s",
             MODELO_LLM, MODELO_EMBED, MODELO_VISION if USAR_VISION else "(off)")
    Settings.llm = Ollama(model=MODELO_LLM, request_timeout=300)
    Settings.embed_model = OllamaEmbedding(model_name=MODELO_EMBED)

    paso("Abriendo base vectorial Chroma")
    log.info("Carpeta DB: %s", os.path.abspath(CARPETA_DB))
    db = chromadb.PersistentClient(path=CARPETA_DB)
    collection = db.get_or_create_collection(COLECCION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    log.info("Vectores ya presentes en la colección: %d", collection.count())

    # Qué documentos ya están indexados (según el metadato 'fuente')
    indexados = set()
    if collection.count():
        datos = collection.get(include=["metadatas"])
        for m in datos.get("metadatas") or []:
            if m and m.get("fuente"):
                indexados.add(m["fuente"])
        log.info("Documentos ya indexados (%d): %s",
                 len(indexados), ", ".join(sorted(indexados)) or "(ninguno)")

    # Índice sobre la base vectorial (funciona vacía o con datos)
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Detectar archivos nuevos en la carpeta y agregarlos incrementalmente
    archivos = listar_archivos(CARPETA_DOCS)
    log.info("Archivos soportados en '%s': %d", CARPETA_DOCS, len(archivos))
    if not archivos and not indexados:
        log.error("No hay archivos (.pdf .txt .docx) ni nada indexado. "
                  "Poné documentos en '%s'.", CARPETA_DOCS)
        sys.exit(1)

    nuevos = [a for a in archivos if a not in indexados]

    if not nuevos:
        paso("Sin archivos nuevos: el índice ya está al día")
    else:
        paso("Indexando %d archivo(s) nuevo(s)" % len(nuevos))
        log.info("Nuevos: %s", ", ".join(nuevos))
        docs = cargar_documentos(CARPETA_DOCS, nuevos)

        paso("Generando embeddings e indexando (esto puede tardar)")
        t = time.time()
        for d in docs:
            log.info("  indexando '%s'...", d.metadata.get("fuente"))
            index.insert(d)
        log.info("Indexación incremental completa en %.1fs", time.time() - t)
        log.info("Vectores en la colección ahora: %d", collection.count())

    paso("Listo. Modo consulta (escribí 'salir' para terminar)")
    qe = index.as_query_engine(similarity_top_k=TOP_K)

    while True:
        try:
            pregunta = input("\nPregunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if pregunta.lower() in ("salir", "exit", "quit"):
            break
        if not pregunta:
            continue
        log.info("Consultando (top_k=%d)...", TOP_K)
        t = time.time()
        resp = qe.query(pregunta)
        log.info("Respuesta generada en %.1fs", time.time() - t)
        print("\n" + str(resp))

    log.info("Hasta luego.")


if __name__ == "__main__":
    main()