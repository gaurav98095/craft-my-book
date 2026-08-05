"""Step 1 — Load Pipeline A's chunks (no re-chunking)."""

import json
from collections import Counter
from typing import Any, Dict, List

from ..ingestion.setup import PATHS
from .setup import PipelineBConfig, toc_log


def load_chunks_from_pipeline_a(cfg: PipelineBConfig) -> List[Dict[str, Any]]:
    """
    Load Stage 4's chunks and their Layer 3 metadata.

    Fails loudly on a mismatch rather than skipping: a chunk in the metadata
    with no file on disk means the corpus was rebuilt without re-chunking, and
    every id downstream would be wrong.
    """
    if not PATHS.chunk_metadata.exists():
        raise FileNotFoundError(
            f"{PATHS.chunk_metadata} not found. Run Stage 4 of Document "
            f"Preprocessing first - Pipeline B does not chunk."
        )

    metadata = json.loads(PATHS.chunk_metadata.read_text(encoding="utf-8"))
    chunks: List[Dict[str, Any]] = []
    missing: List[str] = []

    for chunk_id, record in metadata.items():
        path = PATHS.chunks / f"{chunk_id}.txt"
        if not path.exists():
            missing.append(chunk_id)
            continue
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": path.read_text(encoding="utf-8"),
                "source_document": record.get("source_document"),
                "source_type": record.get("source_type"),
                "timestamp": record.get("timestamp"),
                "figures": record.get("figures", []),
                "token_count": record.get("token_count", 0),
            }
        )

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} chunks are in chunk_metadata.json but not on disk "
            f"(e.g. {missing[:3]}). The corpus and the chunks are out of step - "
            f"re-run Stage 4."
        )

    chunks.sort(key=lambda c: c["chunk_id"])

    if cfg.quick_test_mode:
        chunks = chunks[: cfg.quick_test_chunks]
        toc_log.warning(
            f"QUICK TEST MODE: using only {len(chunks)} chunks. "
            f"The resulting TOC is not a real TOC."
        )

    by_type = Counter(c["source_type"] for c in chunks)
    total_tokens = sum(c["token_count"] for c in chunks)

    toc_log.info(f"Loaded {len(chunks)} chunks from {PATHS.chunks}")
    toc_log.info(f"  corpus tokens : {total_tokens:,}")
    toc_log.info(f"  by source type: {dict(by_type)}")
    toc_log.info(f"  with figures  : {sum(1 for c in chunks if c['figures'])}")
    toc_log.info(f"  with timestamps: {sum(1 for c in chunks if c['timestamp'])}")

    return chunks
