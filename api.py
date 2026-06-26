#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API HTTP para consultar el RAG local desde cualquier lado.

Expone un endpoint que, en cada pregunta:
    recibe la pregunta -> busca en Chroma -> se la pasa a Ollama -> responde.

Reutiliza la MISMA base vectorial (chroma_db) y la MISMA config de modelos
que contexto.py (única fuente de verdad). El índice y el motor de consulta
se arman una sola vez al arrancar; cada request solo hace la búsqueda.

Cómo correr:
    venv\\Scripts\\activate
    py api.py
        (equivale a: uvicorn api:app --host 0.0.0.0 --port 8000)

Endpoints:
    GET  /salud       -> estado y cantidad de vectores
    POST /preguntar   -> {"pregunta": "...", "top_k": 4 (opcional)}

IMPORTANTE: la API NO reindexa. Para cargar documentos nuevos, corré
contexto.py (indexación incremental) y después reiniciá esta API.
"""

import logging
import os

import chromadb
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Config compartida con el script principal (modelos, rutas, TOP_K)
from contexto import CARPETA_DB, COLECCION, MODELO_LLM, MODELO_EMBED, TOP_K

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag-api")

# ----------------------------------------------------------------------------
# Configuración de modelos Ollama
# ----------------------------------------------------------------------------
Settings.llm = Ollama(model=MODELO_LLM, request_timeout=300)
Settings.embed_model = OllamaEmbedding(model_name=MODELO_EMBED)

# ----------------------------------------------------------------------------
# Base vectorial + índice (se abre una sola vez al levantar el servidor)
# ----------------------------------------------------------------------------
db = chromadb.PersistentClient(path=CARPETA_DB)
collection = db.get_or_create_collection(COLECCION)
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store)
log.info("RAG listo. LLM=%s  Embeddings=%s  Vectores=%d",
         MODELO_LLM, MODELO_EMBED, collection.count())

app = FastAPI(title="ollamaRAG API", version="1.0")


class Consulta(BaseModel):
    pregunta: str


@app.get("/")
def home():
    """Sirve la interfaz de chat (chat.html, al lado de este archivo)."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "chat.html"))


@app.get("/salud")
def salud():
    """Chequeo rápido de que la API y la base están vivas."""
    return {
        "estado": "ok",
        "modelo_llm": MODELO_LLM,
        "modelo_embed": MODELO_EMBED,
        "vectores": collection.count(),
    }


@app.post("/preguntar")
def preguntar(c: Consulta):
    """RAG: busca contexto en Chroma y deja que Ollama redacte la respuesta."""
    log.info("Pregunta (top_k=%d): %s", TOP_K, c.pregunta)
    qe = index.as_query_engine(similarity_top_k=TOP_K)
    resp = qe.query(c.pregunta)

    # Qué trozos/documentos usó para responder (trazabilidad)
    fuentes = []
    for n in getattr(resp, "source_nodes", []):
        fuentes.append({
            "fuente": n.node.metadata.get("fuente"),
            "score": round(n.score, 4) if n.score is not None else None,
        })

    return {"respuesta": str(resp), "fuentes": fuentes}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
