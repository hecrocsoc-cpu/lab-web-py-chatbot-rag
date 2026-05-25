"""
chatbot.py — Lógica RAG: recupera fragmentos relevantes de ChromaDB
y genera respuestas usando Groq como LLM.
"""

import os
import re
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────────
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "barco_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"         # modelo rápido y gratuito en Groq
N_FRAGMENTOS = 5                        # fragmentos a recuperar por consulta
MAX_HISTORIAL = 6                       # turnos de conversación a mantener

# Patrones básicos de datos personales
PATRON_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PATRON_TELEFONO = re.compile(r"\b(\+?\d[\d\s\-\(\)]{7,})\b")
PATRON_DNI = re.compile(r"\b\d{8}[A-HJ-NP-TV-Z]\b", re.IGNORECASE)

SYSTEM_PROMPT = """Eres un asistente experto en mantenimiento náutico.
Responde ÚNICAMENTE basándote en el contexto de documentos proporcionado.
Si el contexto no contiene información suficiente para responder la pregunta, di exactamente:
"No tengo información sobre eso en los documentos disponibles."
Usa toda la información del contexto que sea relevante, aunque sea parcial.
No inventes datos, cifras ni recomendaciones que no aparezcan en el contexto.
Responde en español, de forma clara y práctica.
Cuando sea relevante, indica de qué documento proviene la información."""

# ── Inicialización (se hace una sola vez al importar) ──────────────────────────
print("🧠 Cargando modelo de embeddings...")
_modelo_embeddings = SentenceTransformer(EMBEDDING_MODEL)

print("💾 Conectando a ChromaDB...")
_chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
_coleccion = _chroma_client.get_collection(COLLECTION_NAME)

print("🤖 Inicializando cliente Groq...")
_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Historial en memoria: { session_id: [{"role": ..., "content": ...}] }
_historiales: dict[str, list[dict]] = {}


# ── Utilidades ─────────────────────────────────────────────────────────────────

def detectar_datos_personales(texto: str) -> list[str]:
    """Devuelve lista de tipos de datos personales detectados en el texto."""
    encontrados = []
    if PATRON_EMAIL.search(texto):
        encontrados.append("email")
    if PATRON_TELEFONO.search(texto):
        encontrados.append("teléfono")
    if PATRON_DNI.search(texto):
        encontrados.append("DNI/NIF")
    return encontrados


def recuperar_fragmentos(pregunta: str, n: int = N_FRAGMENTOS) -> list[dict]:
    """Busca los N fragmentos más relevantes en ChromaDB."""
    embedding = _modelo_embeddings.encode([pregunta]).tolist()
    resultados = _coleccion.query(
        query_embeddings=embedding,
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    fragmentos = []
    for doc, meta, dist in zip(
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0],
    ):
        fragmentos.append({
            "texto": doc,
            "fuente": meta["fuente"],
            "similitud": round(1 - dist, 3),  # cosine distance → similitud
        })

    return fragmentos


def construir_contexto(fragmentos: list[dict]) -> str:
    """Formatea los fragmentos como contexto para el prompt."""
    partes = []
    for i, f in enumerate(fragmentos, 1):
        partes.append(f"[Fragmento {i} — {f['fuente']}]\n{f['texto']}")
    return "\n\n---\n\n".join(partes)


# ── Función principal ──────────────────────────────────────────────────────────

def chat(pregunta: str, session_id: str) -> dict:
    """
    Responde una pregunta usando RAG:
    1. Recupera los N fragmentos más relevantes de ChromaDB.
    2. Construye el prompt con contexto + historial.
    3. Llama a Groq para generar la respuesta.
    4. Devuelve respuesta, fuentes y metadatos.
    """

    # Detección de datos personales
    datos_personales = detectar_datos_personales(pregunta)
    advertencia = None
    if datos_personales:
        tipos = ", ".join(datos_personales)
        advertencia = (
            f"⚠️ Tu pregunta parece contener {tipos}. "
            "Esta información se enviará al modelo de lenguaje. "
            "Considera reformular la pregunta sin incluir datos personales."
        )

    # Recuperar fragmentos relevantes
    fragmentos = recuperar_fragmentos(pregunta)
    contexto = construir_contexto(fragmentos)
    fuentes = list({f["fuente"] for f in fragmentos})  # sin duplicados

    # Construir mensajes para Groq
    if session_id not in _historiales:
        _historiales[session_id] = []

    historial = _historiales[session_id]

    # Mensaje del usuario con el contexto incrustado
    mensaje_usuario = f"""Contexto de documentos:
{contexto}

Pregunta: {pregunta}"""

    messages_para_groq = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + historial[-MAX_HISTORIAL:]          # últimos N turnos
        + [{"role": "user", "content": mensaje_usuario}]
    )

    # Llamada a Groq
    respuesta_groq = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages_para_groq,
        temperature=0.2,          # respuestas más deterministas
        max_tokens=800,
    )
    respuesta_texto = respuesta_groq.choices[0].message.content

    # Actualizar historial (solo pregunta + respuesta, sin el contexto completo)
    historial.append({"role": "user", "content": pregunta})
    historial.append({"role": "assistant", "content": respuesta_texto})

    resultado = {
        "respuesta": respuesta_texto,
        "fuentes": fuentes,
        "session_id": session_id,
        "fragmentos_usados": len(fragmentos),
    }
    if advertencia:
        resultado["advertencia_privacidad"] = advertencia

    return resultado


def get_historial(session_id: str) -> list[dict]:
    """Devuelve el historial de una sesión."""
    return _historiales.get(session_id, [])


def listar_documentos() -> list[str]:
    """Devuelve los nombres únicos de documentos indexados."""
    todos = _coleccion.get(include=["metadatas"])
    fuentes = {m["fuente"] for m in todos["metadatas"]}
    return sorted(fuentes)
