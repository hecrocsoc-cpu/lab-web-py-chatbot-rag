"""
api.py — API FastAPI que expone el chatbot RAG.

Endpoints:
  POST /chat                        → pregunta → respuesta + fuentes
  GET  /chat/history/{session_id}   → historial de la sesión
  GET  /documentos                  → lista de documentos indexados

Seguridad implementada:
  - Rate limiting: máx 10 peticiones/min por IP
  - Validación de input: longitud máxima 500 caracteres
  - Logging de cada llamada (sin contenido de documentos)
  - Detección de datos personales (en chatbot.py)
"""

import logging
import time
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

import chatbot

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Rate limiting (en memoria; para producción usar Redis) ─────────────────────
# Estructura: { ip: [timestamp1, timestamp2, ...] }
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_MAX = 10        # peticiones máximas
RATE_LIMIT_WINDOW = 60     # segundos


def check_rate_limit(ip: str) -> None:
    """Lanza HTTPException 429 si la IP supera el rate limit."""
    ahora = time.time()
    ventana_inicio = ahora - RATE_LIMIT_WINDOW

    # Limpiar timestamps fuera de la ventana
    _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if t > ventana_inicio]

    if len(_rate_limit_store[ip]) >= RATE_LIMIT_MAX:
        logger.warning(f"Rate limit superado para IP: {ip}")
        raise HTTPException(
            status_code=429,
            detail=f"Demasiadas peticiones. Máximo {RATE_LIMIT_MAX} por minuto.",
        )

    _rate_limit_store[ip].append(ahora)


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Chatbot RAG — Mantenimiento Náutico",
    description="Responde preguntas sobre mantenimiento de barcos usando RAG sobre documentos propios.",
    version="1.0.0",
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class PreguntaRequest(BaseModel):
    pregunta: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Pregunta del usuario (máximo 500 caracteres)",
        examples=["¿Cada cuánto hay que cambiar el impeller?"],
    )
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Identificador único de sesión",
        examples=["usuario_123"],
    )


class RespuestaResponse(BaseModel):
    respuesta: str
    fuentes: list[str]
    session_id: str
    fragmentos_usados: int
    advertencia_privacidad: str | None = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=RespuestaResponse, summary="Hacer una pregunta al chatbot")
async def endpoint_chat(request: Request, body: PreguntaRequest):
    """
    Recibe una pregunta y devuelve una respuesta basada en los documentos indexados.
    Mantiene historial de conversación por session_id.
    """
    ip = request.client.host

    # Rate limiting
    check_rate_limit(ip)

    # Logging (sin loguear el contenido de documentos)
    logger.info(
        f"POST /chat | IP={ip} | session={body.session_id} | "
        f"pregunta_len={len(body.pregunta)}"
    )

    try:
        resultado = chatbot.chat(body.pregunta, body.session_id)
    except Exception as e:
        logger.error(f"Error en chatbot.chat: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar la pregunta.")

    logger.info(
        f"Respuesta generada | session={body.session_id} | "
        f"fuentes={resultado['fuentes']} | fragmentos={resultado['fragmentos_usados']}"
    )

    return resultado


@app.get(
    "/chat/history/{session_id}",
    summary="Obtener historial de una sesión",
    response_model=list[dict],
)
async def endpoint_historial(request: Request, session_id: str):
    """Devuelve el historial de mensajes de una sesión."""
    ip = request.client.host
    check_rate_limit(ip)

    logger.info(f"GET /chat/history/{session_id} | IP={ip}")
    historial = chatbot.get_historial(session_id)

    if not historial:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró historial para la sesión '{session_id}'.",
        )

    return historial


@app.get("/documentos", summary="Listar documentos indexados", response_model=list[str])
async def endpoint_documentos(request: Request):
    """Devuelve la lista de documentos que están indexados en ChromaDB."""
    ip = request.client.host
    check_rate_limit(ip)

    logger.info(f"GET /documentos | IP={ip}")
    return chatbot.listar_documentos()


@app.get("/", include_in_schema=False)
async def raiz():
    return {"mensaje": "API activa. Accede a /docs para la documentación interactiva."}
