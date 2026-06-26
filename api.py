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
import threading
import uuid

import chromadb
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer
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
# context_window (=> num_ctx en Ollama) tiene que ser amplio: al chatear, el
# prompt lleva el contexto recuperado + el HISTORIAL de la conversación. Si se
# queda corto, el modelo trunca y puede devolver tokens basura.
Settings.llm = Ollama(model=MODELO_LLM, request_timeout=300, context_window=8192)
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

app = FastAPI(title="Guanaco Advisor API", version="2.0")

# ----------------------------------------------------------------------------
# Memoria de conversación por sesión
# ----------------------------------------------------------------------------
# Cada sesión (identificada por session_id) tiene su propio historial de chat,
# guardado en memoria del servidor. Es liviano y suficiente para uso interno.
# NOTA: al reiniciar la API estas conversaciones se pierden (no se persisten).
TOKENS_HISTORIAL = 3000          # cuánto historial recordar (en tokens)
_sesiones = {}                   # session_id -> ChatMemoryBuffer
_lock = threading.Lock()


def get_memoria(session_id: str) -> ChatMemoryBuffer:
    """Devuelve (creándola si hace falta) la memoria de esa sesión."""
    with _lock:
        mem = _sesiones.get(session_id)
        if mem is None:
            mem = ChatMemoryBuffer.from_defaults(token_limit=TOKENS_HISTORIAL)
            _sesiones[session_id] = mem
        return mem


class Consulta(BaseModel):
    pregunta: str
    session_id: str | None = None   # si no llega, se crea una sesión nueva


class Reinicio(BaseModel):
    session_id: str


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
    """RAG con memoria: usa el historial de la sesión + contexto de Chroma."""
    session_id = c.session_id or uuid.uuid4().hex
    memoria = get_memoria(session_id)
    log.info("Pregunta (sesion=%s, top_k=%d): %s",
             session_id[:8], TOP_K, c.pregunta)

    # Chat engine: condensa la pregunta con el historial, recupera contexto y
    # responde teniendo en cuenta lo que se habló antes en esta sesión.
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memoria,
        similarity_top_k=TOP_K,
        system_prompt=(
            "Sos un asistente que responde en español de forma clara y concisa, "
            "basándote en el contexto disponible y en la conversación previa. "
            "Si no tenés información suficiente, decilo."
        ),
    )
    resp = chat_engine.chat(c.pregunta)

    # Qué trozos/documentos usó para responder (trazabilidad)
    fuentes = []
    for n in getattr(resp, "source_nodes", []):
        fuentes.append({
            "fuente": n.node.metadata.get("fuente"),
            "score": round(n.score, 4) if n.score is not None else None,
        })

    return {"respuesta": str(resp), "fuentes": fuentes, "session_id": session_id}


@app.post("/reiniciar")
def reiniciar(r: Reinicio):
    """Olvida el historial de una sesión (arranca una conversación nueva)."""
    with _lock:
        _sesiones.pop(r.session_id, None)
    log.info("Sesión reiniciada: %s", r.session_id[:8])
    return {"estado": "ok", "session_id": r.session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
