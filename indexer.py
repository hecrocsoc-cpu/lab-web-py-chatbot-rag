"""
indexer.py — Lee los documentos de docs/, los fragmenta, crea embeddings
y los guarda en ChromaDB.

Usamos sentence-transformers para los embeddings (local, gratis).
"""

import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────────
DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "barco_docs"
CHUNK_SIZE = 500          # caracteres por fragmento
CHUNK_OVERLAP = 100       # solapamiento entre fragmentos

# Modelo de embeddings local (se descarga automáticamente la primera vez, ~90 MB)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ── Funciones ──────────────────────────────────────────────────────────────────

def leer_documentos(directorio: str) -> list[dict]:
    """Lee todos los .txt del directorio y devuelve lista de {nombre, contenido}."""
    documentos = []
    archivos = glob.glob(os.path.join(directorio, "*.txt"))

    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos .txt en '{directorio}'")

    for ruta in archivos:
        with open(ruta, "r", encoding="utf-8") as f:
            contenido = f.read().strip()
        nombre = os.path.basename(ruta)
        documentos.append({"nombre": nombre, "contenido": contenido})
        print(f"  📄 Leído: {nombre} ({len(contenido)} caracteres)")

    return documentos


def fragmentar(texto: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide el texto en fragmentos de tamaño chunk_size con solapamiento.
    Intenta cortar en saltos de línea para no partir frases a medias.
    """
    chunks = []
    inicio = 0

    while inicio < len(texto):
        fin = inicio + chunk_size

        # Si no es el último fragmento, buscar un salto de línea cercano
        if fin < len(texto):
            corte = texto.rfind("\n", inicio, fin)
            if corte > inicio:
                fin = corte

        chunk = texto[inicio:fin].strip()
        if chunk:
            chunks.append(chunk)

        inicio = fin - overlap  # retroceder para el solapamiento

    return chunks


def indexar():
    """Pipeline completo: leer → fragmentar → embeddings → ChromaDB."""

    print("\n🔍 Leyendo documentos...")
    documentos = leer_documentos(DOCS_DIR)

    print("\n✂️  Fragmentando documentos...")
    todos_los_chunks = []  # {"texto", "doc_nombre", "chunk_id"}
    for doc in documentos:
        chunks = fragmentar(doc["contenido"])
        for i, chunk in enumerate(chunks):
            todos_los_chunks.append({
                "texto": chunk,
                "doc_nombre": doc["nombre"],
                "chunk_id": f"{doc['nombre']}__chunk_{i}",
            })
        print(f"  ✂️  {doc['nombre']}: {len(chunks)} fragmentos")

    print(f"\n📊 Total fragmentos: {len(todos_los_chunks)}")

    # ── Embeddings ──────────────────────────────────────────────────────────────
    print(f"\n🧠 Cargando modelo de embeddings '{EMBEDDING_MODEL}'...")
    modelo = SentenceTransformer(EMBEDDING_MODEL)

    textos = [c["texto"] for c in todos_los_chunks]
    print("⚙️  Calculando embeddings (puede tardar unos segundos)...")
    embeddings = modelo.encode(textos, show_progress_bar=True).tolist()

    # ── ChromaDB ────────────────────────────────────────────────────────────────
    print(f"\n💾 Guardando en ChromaDB ('{CHROMA_DIR}')...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Borrar colección existente para reindexar limpio
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  🗑️  Colección anterior eliminada")
    except Exception:
        pass

    coleccion = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    coleccion.add(
        ids=[c["chunk_id"] for c in todos_los_chunks],
        embeddings=embeddings,
        documents=textos,
        metadatas=[{"fuente": c["doc_nombre"]} for c in todos_los_chunks],
    )

    # ── Resumen ─────────────────────────────────────────────────────────────────
    total_chars = sum(len(c["texto"]) for c in todos_los_chunks)
    total_tokens_aprox = total_chars // 4  # estimación: ~4 chars por token

    print("\n" + "=" * 50)
    print("✅ INDEXACIÓN COMPLETA")
    print(f"   📚 Documentos procesados : {len(documentos)}")
    print(f"   🧩 Fragmentos creados    : {len(todos_los_chunks)}")
    print(f"   🔤 Tokens estimados      : ~{total_tokens_aprox:,}")
    print(f"   💰 Coste estimado        : $0.00 (embeddings locales)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    indexar()
