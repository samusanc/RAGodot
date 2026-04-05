import chromadb
import os
import re
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
CHK_DIR     = "./godot_chk"          # root folder with your .chk files
OLLAMA_URL  = "http://192.168.0.36:11434/api/embeddings"
MODEL_NAME  = "qwen3-embedding:8b"
COLLECTION  = "godot_docs"

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_chk_file(filepath: str) -> list[dict]:
    """Split a .chk file on ---CHUNK--- and return structured chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    raw_chunks = [c.strip() for c in raw.split("---CHUNK---") if c.strip()]
    chunks = []

    for i, chunk in enumerate(raw_chunks):
        lines = chunk.splitlines()

        # Extract metadata from the first few header lines
        doc_name = ""
        doc_path = ""
        summary  = ""
        body_start = 0

        for j, line in enumerate(lines):
            if line.startswith("Document:"):
                doc_name = line.split(":", 1)[1].strip()
            elif line.startswith("Path:"):
                doc_path = line.split(":", 1)[1].strip()
            elif line.startswith("Summary:"):
                summary = line.split(":", 1)[1].strip()
            else:
                body_start = j
                break

        body = "\n".join(lines[body_start:]).strip()

        chunks.append({
            "doc_name": doc_name,
            "doc_path": doc_path,
            "summary":  summary,
            "body":     body,
            "source_file": os.path.basename(filepath),
            "chunk_index": i,
        })

    return chunks


def load_all_chk_files(root_dir: str) -> list[dict]:
    all_chunks = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".chk"):
                fpath = os.path.join(dirpath, fname)
                all_chunks.extend(parse_chk_file(fpath))
    return all_chunks


def make_id(chunk: dict) -> str:
    """Stable unique ID per chunk."""
    safe_path = re.sub(r"[^a-zA-Z0-9_-]", "_", chunk["doc_path"])
    return f"{safe_path}__chunk{chunk['chunk_index']}"


def make_document(chunk: dict) -> str:
    """What gets embedded — combine summary + body for richer retrieval."""
    parts = []
    if chunk["summary"]:
        parts.append(f"Summary: {chunk['summary']}")
    if chunk["body"]:
        parts.append(chunk["body"])
    return "\n\n".join(parts)


# ── Ingest ────────────────────────────────────────────────────────────────────
def ingest(collection, chunks: list[dict]):
    existing = set(collection.get()["ids"])
    
    ids, docs, metas = [], [], []

    for chunk in chunks:
        cid = make_id(chunk)
        if cid in existing:
            continue                          # skip already-loaded chunks

        ids.append(cid)
        docs.append(make_document(chunk))
        metas.append({
            "doc_name":    chunk["doc_name"],
            "doc_path":    chunk["doc_path"],
            "summary":     chunk["summary"],
            "source_file": chunk["source_file"],
            "chunk_index": chunk["chunk_index"],
        })

    if not ids:
        print("Nothing new to ingest.")
        return

    # ChromaDB recommends batches of ≤ 5000
    batch = 256
    for start in range(0, len(ids), batch):
        collection.add(
            ids=ids[start:start+batch],
            documents=docs[start:start+batch],
            metadatas=metas[start:start+batch],
        )
        print(f"  Ingested {min(start+batch, len(ids))}/{len(ids)} chunks…")

    print(f"Done. {len(ids)} new chunks added.")


# ── Query / Test ──────────────────────────────────────────────────────────────
def test_rag(collection, queries: list[str], n_results: int = 3):
    for q in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {q}")
        print('='*60)

        results = collection.query(query_texts=[q], n_results=n_results)

        for rank, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ), start=1):
            print(f"\n[{rank}] {meta['doc_name']}  (distance: {dist:.4f})")
            print(f"    Source : {meta['source_file']}  chunk#{meta['chunk_index']}")
            print(f"    Summary: {meta['summary'][:80]}…")
            print(f"    Body   : {doc[:200]}…")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    qwen_ef = OllamaEmbeddingFunction(model_name=MODEL_NAME, url=OLLAMA_URL)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=qwen_ef,
    )

    # 1. Ingest
    print("Scanning .chk files…")
    chunks = load_all_chk_files(CHK_DIR)
    print(f"Found {len(chunks)} chunks across all .chk files")
    ingest(collection, chunks)

    # 2. Test queries — tweak these to match your docs
    test_rag(collection, [
        "How do I check if two bounding boxes overlap?",
        "What is the AABB position property?",
        "How do I get the volume of an AABB?",
        "What happens if AABB size is negative?",
    ])
