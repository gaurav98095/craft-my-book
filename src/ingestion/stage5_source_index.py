"""Stage 5 — The Source Index."""

import json
import time
import shutil
from typing import Any, Dict

from tqdm.auto import tqdm

from .setup import PATHS
from .stage4_chunking import stage4_log


def build_source_index(
    model_name: str = "all-MiniLM-L6-v2", batch_size: int = 256, rebuild: bool = False
) -> Dict[str, Any]:
    """
    Populate the Chroma collection Layer 3 reads from.

    Skips cleanly if chromadb is not installed, so a corpus can still be
    inspected without it -- but says so loudly, because a missing index is a
    silent quality loss in Pipeline C rather than an error.
    """
    try:
        import chromadb
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )
    except ImportError:
        stage4_log.warning("chromadb / sentence-transformers not installed.")
        stage4_log.warning("  pip install chromadb sentence-transformers")
        stage4_log.warning(
            "  Without data/source_index, SourceMemory.find_related() "
            "returns nothing and every section is written only from "
            "the chunks the TOC assigned it."
        )
        return {"status": "skipped", "reason": "chromadb not installed"}

    if not PATHS.chunk_metadata.exists():
        return {"status": "skipped", "reason": "no chunk_metadata.json - run Stage 4"}

    metadata = json.loads(PATHS.chunk_metadata.read_text(encoding="utf-8"))
    if rebuild and PATHS.source_index.exists():
        shutil.rmtree(PATHS.source_index)

    client = chromadb.PersistentClient(path=str(PATHS.source_index))
    collection = client.get_or_create_collection(
        name="source_chunks",
        embedding_function=SentenceTransformerEmbeddingFunction(model_name=model_name),
    )

    ids, documents, metadatas = [], [], []
    for chunk_id, record in metadata.items():
        path = PATHS.chunks / f"{chunk_id}.txt"
        if not path.exists():
            continue
        ids.append(chunk_id)
        documents.append(path.read_text(encoding="utf-8"))
        # Chroma metadata values must be scalars, so figures are reduced to a
        # count here. The full records stay in chunk_metadata.json.
        metadatas.append(
            {
                "source_document": record["source_document"] or "unknown",
                "source_type": record["source_type"],
                "doc_slug": record.get("doc_slug", ""),
                "token_count": int(record.get("token_count", 0)),
                "figure_count": len(record.get("figures", [])),
            }
        )

    stage4_log.info(f"Embedding {len(ids)} chunks with {model_name} ...")
    t0 = time.time()
    for i in tqdm(range(0, len(ids), batch_size), desc="Embedding chunks"):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    result = {
        "status": "ok",
        "collection": "source_chunks",
        "path": str(PATHS.source_index),
        "chunks_indexed": len(ids),
        "embedding_model": model_name,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    stage4_log.info(
        f"source_index built: {len(ids)} chunks in " f"{result['elapsed_seconds']}s"
    )
    return result
