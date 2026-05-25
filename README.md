# 🚢 Chatbot RAG — Plan de Mantenimiento Náutico

Chatbot con **Retrieval-Augmented Generation (RAG)** que responde preguntas sobre mantenimiento náutico basándose en documentos propios. El sistema recupera fragmentos relevantes de una base de datos vectorial y genera respuestas precisas usando un LLM, sin inventar información.

**Tema elegido:** Plan de mantenimiento del Buque Guardamar Talía y documentación náutica general.

---

## 🏗️ Arquitectura

```
docs/                        ← documentos fuente (.txt)
indexer.py                   ← lee docs, crea embeddings, guarda en ChromaDB
chatbot.py                   ← lógica RAG: recupera fragmentos y responde con Groq
api.py                       ← API REST con FastAPI
chroma_db/                   ← base de datos vectorial (se genera al indexar)
```

### Flujo RAG

```
Pregunta → Embedding → ChromaDB → Top 5 fragmentos → Groq LLM → Respuesta + fuentes
```

---

## 🛠️ Stack tecnológico

| Componente | Tecnología |
|---|---|
| LLM | Groq (llama-3.1-8b-instant) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) — local, sin coste |
| Base de datos vectorial | ChromaDB |
| API | FastAPI + Uvicorn |
| Dependencias | Python 3.12, python-dotenv, pydantic |

---

## 📄 Documentos indexados

- `plan_mantenimiento_guardamar_talia_limpio.txt` — Plan de mantenimiento real del Buque Guardamar Talía (motores MTU, reductoras ZF, servo-timón, generadores, etc.)
- `motor_propulsion.txt` — Sistemas de propulsión náutica
- `casco_obra_viva.txt` — Mantenimiento de casco y obra viva
- `electricidad_electronica.txt` — Sistemas eléctricos y electrónicos
- `velas_jarcia.txt` — Velas, jarcia y aparejo
- `seguridad_equipamiento.txt` — Seguridad y equipamiento de a bordo

---

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/hecrocsoc-cpu/lab-web-py-chatbot-rag
cd lab-web-py-chatbot-rag
```

### 2. Crear y activar el entorno virtual

```bash
python -m venv venv

# Windows (Git Bash)
source venv/Scripts/activate

# Mac / Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> ⚠️ Si aparece el error `Client.__init__() got an unexpected keyword argument 'proxies'`, ejecuta:
> ```bash
> pip install httpx==0.27.2
> ```

### 4. Configurar la API key de Groq

Copia el archivo de ejemplo y añade tu clave:

```bash
cp .env.example .env
```

Edita `.env` y sustituye con tu API key de [console.groq.com](https://console.groq.com):

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
```

---

## 🗂️ Indexar los documentos

Ejecuta el indexador para procesar los documentos y guardarlos en ChromaDB:

```bash
python indexer.py
```

Deberías ver algo como:

```
📚 Documentos procesados : 6
🧩 Fragmentos creados    : 122
✅ INDEXACIÓN COMPLETA
```

> Cada vez que añadas o modifiques documentos en `docs/`, vuelve a ejecutar `indexer.py` y reinicia el servidor.

---

## 🚀 Arrancar la API

```bash
python -m uvicorn api:app --reload
```

La API estará disponible en `http://127.0.0.1:8000`.

Documentación interactiva (Swagger UI): `http://127.0.0.1:8000/docs`

---

## 📡 Endpoints

### `POST /chat`

Envía una pregunta y obtiene una respuesta basada en los documentos.

**Body:**
```json
{
  "session_id": "mi_sesion",
  "pregunta": "¿Qué mantenimiento diario hay que hacer en los motores principales?"
}
```

**Respuesta:**
```json
{
  "respuesta": "Según el plan de mantenimiento, los motores principales requieren diariamente: comprobar nivel de aceite, comprobar estanqueidad y estado general...",
  "fuentes": ["plan_mantenimiento_guardamar_talia_limpio.txt"],
  "session_id": "mi_sesion",
  "fragmentos_usados": 5,
  "advertencia_privacidad": null
}
```

---

### `GET /chat/history/{session_id}`

Devuelve el historial de conversación de una sesión.

```
GET http://127.0.0.1:8000/chat/history/mi_sesion
```

---

### `GET /documentos`

Lista los documentos indexados disponibles.

```
GET http://127.0.0.1:8000/documentos
```

---

## 🔒 Medidas de privacidad y seguridad

- **Rate limiting:** máximo 10 peticiones por minuto por IP
- **Validación de input:** longitud máxima de pregunta 500 caracteres
- **Logging:** se registra cada llamada sin loguear el contenido de los documentos
- **Detección de datos personales:** si la pregunta contiene email, teléfono o DNI, el sistema advierte antes de enviar al LLM

---

## ⚠️ Limitaciones conocidas

- El sistema responde **únicamente** con información de los documentos indexados. Si la información no está en los docs, responde "No tengo información sobre eso en los documentos disponibles."
- Las preguntas con terminología muy técnica o codificada (como códigos de mantenimiento) pueden no recuperar los fragmentos correctos. Se recomienda formular las preguntas en lenguaje natural.
- El historial de conversación se almacena **en memoria** y se pierde al reiniciar el servidor.
