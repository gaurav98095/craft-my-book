"""
Phase 2.1 — Layer 2: The Draft Store.

Every finished section, on disk as markdown and in a vector index as an
abstract. The index is what lets the Continuity Gate ask "is this plan too
close to something we already wrote?" before a word is generated.
"""

from pathlib import Path
from typing import Dict, List, Optional

from .setup import BookPaths, writer_log


class DraftStore:
    """Layer 2. Append-only: the book so far."""

    def __init__(self, paths: BookPaths, model_name: str = "all-MiniLM-L6-v2"):
        self.sections_dir = paths.sections
        self.sections_dir.mkdir(parents=True, exist_ok=True)
        self.collection = None
        self._index_path = paths.draft_index

        try:
            import chromadb
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )

            client = chromadb.PersistentClient(path=str(self._index_path))
            self.collection = client.get_or_create_collection(
                name="written_sections",
                embedding_function=SentenceTransformerEmbeddingFunction(
                    model_name=model_name
                ),
            )
            writer_log.info(f"Layer 2: draft index at {self._index_path}")
        except ImportError:
            # Not fatal, but it disables the repetition detector, so it is loud.
            writer_log.warning("chromadb/sentence-transformers not installed.")
            writer_log.warning(
                "  Sections will still be written to disk, but the "
                "Continuity Gate's repetition check and the Context "
                "Assembler's 'nearest by meaning' neighbours are DISABLED."
            )

    # ----------------------------------------------------------------- write --
    @staticmethod
    def render_markdown(title: str, content: str) -> str:
        """The one canonical on-disk form of a section."""
        return f"## {title}\n\n{content.strip()}\n"

    def write_markdown(self, section_id: str, title: str, content: str) -> Path:
        """
        Put the section on disk.

        Called by the orchestrator BEFORE the Archivist runs, so a section
        survives even if cataloguing fails. `add_section` re-renders the same
        bytes, so the two writers can never disagree about what the book says.
        """
        path = self.sections_dir / f"{section_id}.md"
        path.write_text(self.render_markdown(title, content), encoding="utf-8")
        return path

    def add_section(
        self, section_id: str, title: str, chapter_id: str, content: str, abstract: str
    ) -> None:
        self.write_markdown(section_id, title, content)
        if self.collection is None:
            return
        try:
            self.collection.upsert(
                ids=[section_id],
                documents=[abstract or title],
                metadatas=[
                    {
                        "section_id": section_id,
                        "title": title,
                        "chapter_id": chapter_id,
                        "word_count": len(content.split()),
                    }
                ],
            )
        except Exception as exc:
            writer_log.error(f"Could not index {section_id}: {exc}")

    # ------------------------------------------------------------------ read --
    def get_full(self, section_id: Optional[str]) -> str:
        if not section_id:
            return ""
        path = self.sections_dir / f"{section_id}.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def get_tail(self, section_id: Optional[str], n_chars: int = 800) -> str:
        """
        The previous section's last N characters, verbatim.

        The cheapest trick in the design: injecting this under "here is how
        the previous section ended, continue from this voice" does more for
        perceived flow than any amount of style-guide text.
        """
        return self.get_full(section_id)[-n_chars:]

    def written_ids(self) -> List[str]:
        return sorted(p.stem for p in self.sections_dir.glob("sec_*.md"))

    def find_similar(
        self, query: str, k: int = 3, exclude: Optional[str] = None
    ) -> List[Dict]:
        """Nearest written sections by meaning. Empty list if the index is off."""
        if self.collection is None or not query.strip():
            return []
        try:
            n_written = self.collection.count()
            if n_written == 0:
                return []
            res = self.collection.query(
                query_texts=[query], n_results=min(k + 1, n_written)
            )
        except Exception as exc:
            writer_log.warning(f"draft index query failed: {exc}")
            return []

        out = []
        for i, sid in enumerate(res["ids"][0]):
            if sid == exclude:
                continue
            distance = res["distances"][0][i] if res.get("distances") else 0.0
            out.append(
                {
                    "section_id": sid,
                    "title": res["metadatas"][0][i].get("title", sid),
                    "abstract": res["documents"][0][i],
                    "similarity": 1 - distance,
                }
            )
        return out[:k]
