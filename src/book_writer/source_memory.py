"""
Phase 2.2 — Layer 3: Source Memory.

The grounding layer: chunks, their provenance, and -- importantly -- an
account of which ones have already been used.

Everything here was produced by Pipeline A. Layer 3 is read-mostly: it opens
chunk_metadata.json and the source_index Chroma collection and never writes
to either.
"""

import json
from typing import Any, Dict, List, Optional

from ..ingestion.setup import PATHS
from .setup import writer_log
from .ledger import BookLedger


class SourceMemory:
    """Layer 3."""

    def __init__(self, ledger: BookLedger, neighbourhood: int = 1):
        self.ledger = ledger
        self.chunks_dir = PATHS.chunks
        self.neighbourhood = neighbourhood

        if not PATHS.chunk_metadata.exists():
            raise FileNotFoundError(
                f"{PATHS.chunk_metadata} not found — run Pipeline A's Stage 4."
            )
        self.meta: Dict[str, Any] = json.loads(
            PATHS.chunk_metadata.read_text(encoding="utf-8")
        )

        self.collection = None
        try:
            import chromadb
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )

            client = chromadb.PersistentClient(path=str(PATHS.source_index))
            self.collection = client.get_collection(
                name="source_chunks",
                embedding_function=SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                ),
            )
            writer_log.info(f"Layer 3: {len(self.meta)} chunks, source index ON")
        except Exception as exc:
            writer_log.warning(
                f"source_index unavailable ({type(exc).__name__}). "
                f"Semantic source expansion is DISABLED — every section "
                f"will be written only from the chunks the TOC assigned it. "
                f"Run Pipeline A's Stage 5 to build it."
            )

    # ------------------------------------------------------------- provenance --
    def _header(self, chunk_id: str) -> str:
        m = self.meta.get(chunk_id, {})
        bits = [
            f"--- SOURCE {chunk_id}",
            f"from: {m.get('source_document', 'unknown')}",
            f"type: {m.get('source_type', 'text')}",
        ]
        ts = m.get("timestamp")
        if ts:
            mins = int(ts["start_seconds"] // 60)
            secs = int(ts["start_seconds"] % 60)
            bits.append(
                f"at ~{mins:d}:{secs:02d}"
                + (" (approx)" if ts.get("approximate") else "")
            )
        header = " | ".join(bits) + " ---\n"

        # The fifteen characters of logic that close failure #1 at its root.
        prior = self.ledger.chunk_already_used(chunk_id)
        if prior:
            header += (
                f"[NOTE: already used in {', '.join(prior)} — "
                f"do not re-explain, build on it]\n"
            )
        return header

    def _read(self, chunk_id: str) -> Optional[str]:
        path = self.chunks_dir / f"{chunk_id}.txt"
        return path.read_text(encoding="utf-8") if path.exists() else None

    # ------------------------------------------------------- 1. assigned source --
    def get_assigned(self, chunk_ids: List[str], expand: bool = True) -> str:
        """
        The chunks the TOC assigned, with provenance and usage notes.

        Chunks arrive with their neighbours: "a chunk boundary is an artifact
        of the chunker, not of the argument, and there is no reason to hand
        the Writer a paragraph that starts mid-thought."
        """
        parts: List[str] = []
        emitted: set = set()

        for cid in chunk_ids:
            body = self._read(cid)
            if body is None:
                parts.append(f"\n[MISSING SOURCE: {cid}]\n")
                continue
            parts.append("\n" + self._header(cid) + body)
            emitted.add(cid)

            if not expand:
                continue
            # neighbours, same document only, without repeating anything
            number = int(cid.split("_")[1])
            home = self.meta.get(cid, {}).get("source_document")
            for n in range(
                number - self.neighbourhood, number + self.neighbourhood + 1
            ):
                nid = f"chunk_{n:04d}"
                if nid in emitted or nid == cid or nid not in self.meta:
                    continue
                if self.meta[nid].get("source_document") != home:
                    continue
                nbody = self._read(nid)
                if nbody:
                    parts.append(
                        f"\n--- CONTEXT {nid} (neighbour of {cid}) ---\n{nbody}"
                    )
                    emitted.add(nid)

        return "".join(parts)

    # --------------------------------------------------- 2. semantic expansion --
    def find_related(self, query: str, exclude: List[str], k: int = 3) -> str:
        """Relevant material the TOC did not assign. Closes coverage gaps."""
        if self.collection is None or not query.strip():
            return ""
        try:
            res = self.collection.query(query_texts=[query], n_results=k + len(exclude))
        except Exception as exc:
            writer_log.warning(f"source index query failed: {exc}")
            return ""

        parts, taken = [], 0
        for i, cid in enumerate(res["ids"][0]):
            if cid in exclude or taken >= k:
                continue
            parts.append(
                f"\n--- RELATED SOURCE {cid} (supporting, not assigned) "
                f"| from: {self.meta.get(cid, {}).get('source_document', '?')} ---\n"
                f"{res['documents'][0][i]}"
            )
            taken += 1
        return "".join(parts)

    # ------------------------------------------------------------- 3. figures --
    def get_figures(self, chunk_ids: List[str]) -> List[Dict]:
        """
        The figures that belong with these chunks.

        Because the Writer is a multimodal model, the Context Assembler can
        attach the image itself next to its description -- the model reads
        the diagram instead of reading somebody's summary of it.
        """
        figures = []
        for cid in chunk_ids:
            for f in self.meta.get(cid, {}).get("figures", []):
                if f.get("description"):
                    figures.append({**f, "chunk_id": cid})
        return figures

    def all_chunk_ids(self) -> List[str]:
        return sorted(self.meta.keys())
