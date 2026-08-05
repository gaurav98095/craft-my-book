"""Step 2 — Fine-grained tag AND relationship extraction."""

from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from .setup import PipelineBConfig, toc_log
from .llm import BookLLM
from .checkpoints import StepCheckpoints

EXTRACTION_SCHEMA = {
    "tags": ["string - a specific, atomic concept"],
    "relationships": [
        {
            "from": "string - a tag",
            "to": "string - a tag",
            "type": "requires|part_of|contrasts_with",
        }
    ],
    "chunk_summary": "string - one sentence on what this chunk teaches",
}


class FineGrainedTagExtractor:
    """
    Extracts fine-grained atomic concepts AND the relationships between them.

    The tag half of the system prompt is preserved verbatim from the working
    implementation -- the bad/good examples are what keep the model off broad
    categories, and they were doing their job.
    """

    def __init__(self, llm: BookLLM, cfg: PipelineBConfig):
        self.llm = llm
        self.cfg = cfg
        self.stats = {"chunks": 0, "failed": 0, "tags": 0, "relationships": 0}

        self.system_prompt = f"""You are an expert knowledge engineer extracting SPECIFIC, ATOMIC concepts from educational content.

YOUR TASK: Extract fine-grained concepts that a student would need to learn, and the relationships between them. Think like an index at the back of a textbook.

CRITICAL RULES FOR GOOD TAGS:
1. Extract SPECIFIC concepts, not broad categories
2. Each tag should be a single learnable concept
3. Prefer concrete over abstract
4. Include techniques, algorithms, parameters, components
5. Extract {cfg.min_tags_per_chunk}-{cfg.max_tags_per_chunk} tags per chunk (more is better than fewer)

EXAMPLES OF BAD TAGS (too broad - NEVER use these):
- "machine learning" - too broad, extract specific ML concepts instead
- "deep learning" - too broad, extract specific DL concepts instead
- "neural networks" - too broad, extract specific NN concepts instead
- "CNN" - too broad, extract CNN components instead
- "NLP" - too broad, extract specific NLP techniques instead
- "optimization" - too broad, extract specific optimization methods instead

EXAMPLES OF GOOD TAGS (specific - USE these):
- "gradient descent", "learning rate", "batch size", "momentum"
- "convolutional layer", "max pooling", "stride", "padding", "kernel size"
- "backpropagation", "chain rule", "weight initialization", "Xavier initialization"
- "dropout regularization", "L2 regularization", "early stopping"
- "attention mechanism", "self-attention", "multi-head attention", "positional encoding"
- "LSTM cell", "forget gate", "input gate", "hidden state"

WHAT TO EXTRACT:
- Algorithms and methods (specific ones, not categories)
- Mathematical concepts (loss functions, activation functions, etc.)
- Architecture components (layers, gates, cells, etc.)
- Hyperparameters and settings (learning rate, batch size, etc.)
- Techniques and tricks (dropout, batch norm, residual connections, etc.)

WHAT NOT TO EXTRACT:
- Broad field names (ML, DL, AI, NLP, CV)
- Vague concepts ("training", "model", "data", "performance")
- Implementation details ("Python", "TensorFlow", "GPU")
- Meta-concepts ("introduction", "overview", "basics", "advanced")

RELATIONSHIPS -- this is the part that decides teaching order:
Report up to {cfg.max_relationships_per_chunk} relationships between the tags you extracted.
  "requires"        -- 'from' CANNOT be understood without 'to' first.
                       Example: {{"from": "transformers", "to": "self-attention", "type": "requires"}}
  "part_of"         -- 'from' is a component of 'to'.
                       Example: {{"from": "forget gate", "to": "LSTM cell", "type": "part_of"}}
  "contrasts_with"  -- the two are alternatives a reader may confuse.
                       Example: {{"from": "batch norm", "to": "layer norm", "type": "contrasts_with"}}

Only report a relationship if THIS TEXT supports it. Do not invent prerequisites from
general knowledge. Both 'from' and 'to' must be tags you extracted above.

Quality over quantity, but do not be too conservative."""

    def _user_message(self, chunk: Dict[str, Any]) -> str:
        return f"""Extract specific, atomic concepts and their relationships from this text chunk.

TEXT CHUNK (ID: {chunk['chunk_id']}):
\"\"\"
{chunk['text']}
\"\"\"

Remember:
- Extract {self.cfg.min_tags_per_chunk}-{self.cfg.max_tags_per_chunk} SPECIFIC concepts (not broad categories)
- Think: "What specific things would a student learn from this?"
- Then report which of those concepts REQUIRE which others."""

    def _absorb(
        self,
        chunk: Dict[str, Any],
        reply: Optional[Dict],
        chunk_tags: Dict,
        relationships: List,
        summaries: Dict,
    ) -> None:
        """Fold one model reply into the accumulating results."""
        self.stats["chunks"] += 1

        if not reply:
            self.stats["failed"] += 1
            chunk_tags[chunk["chunk_id"]] = []
            return

        tags = [
            str(t).strip().lower()
            for t in reply.get("tags", [])
            if isinstance(t, str) and t.strip()
        ]
        tags = list(dict.fromkeys(tags))  # dedupe, keep order
        chunk_tags[chunk["chunk_id"]] = tags
        summaries[chunk["chunk_id"]] = str(reply.get("chunk_summary", "")).strip()
        self.stats["tags"] += len(tags)

        # A relationship is only kept if BOTH endpoints are tags this chunk
        # actually extracted. Models will otherwise happily relate concepts that
        # appear nowhere in the text, and those invented prerequisites would
        # end up in the Book Ledger as facts about the material.
        tagset = set(tags)
        for rel in reply.get("relationships", []):
            if not isinstance(rel, dict):
                continue
            src = str(rel.get("from", "")).strip().lower()
            dst = str(rel.get("to", "")).strip().lower()
            kind = str(rel.get("type", "")).strip().lower()
            if (
                src in tagset
                and dst in tagset
                and src != dst
                and kind in ("requires", "part_of", "contrasts_with")
            ):
                relationships.append(
                    {
                        "from": src,
                        "to": dst,
                        "type": kind,
                        "chunk_id": chunk["chunk_id"],
                    }
                )
                self.stats["relationships"] += 1

    def extract_all(
        self,
        chunks: List[Dict[str, Any]],
        ckpt: Optional[StepCheckpoints] = None,
        partial: Optional[Dict] = None,
    ) -> Tuple[Dict[str, List[str]], List[Dict], Dict[str, str]]:
        """
        Returns (chunk_tags, relationships, chunk_summaries).

        Generation is sequential -- one forward pass per chunk -- so this is the
        long pole of the whole pipeline. Partial results are written every
        `extraction_checkpoint_every` chunks and picked up on the next run, so an
        interrupted job resumes from where it stopped instead of re-paying for
        every call it already made.
        """
        chunk_tags: Dict[str, List[str]] = dict((partial or {}).get("chunk_tags", {}))
        relationships: List[Dict] = list((partial or {}).get("relationships", []))
        summaries: Dict[str, str] = dict((partial or {}).get("summaries", {}))

        todo = [c for c in chunks if c["chunk_id"] not in chunk_tags]
        if len(todo) < len(chunks):
            toc_log.info(
                f"  resuming: {len(chunks) - len(todo)} chunks already done, "
                f"{len(todo)} remaining"
            )

        for done, chunk in enumerate(
            tqdm(todo, desc="Extracting tags + relationships"), 1
        ):
            reply = self.llm.generate_structured(
                self.system_prompt,
                self._user_message(chunk),
                EXTRACTION_SCHEMA,
                max_tokens=self.cfg.max_tokens_extraction,
                temperature=0.2,
            )
            self._absorb(chunk, reply, chunk_tags, relationships, summaries)

            if ckpt and done % self.cfg.extraction_checkpoint_every == 0:
                ckpt.save(
                    "step2_extraction_partial",
                    {
                        "chunk_tags": chunk_tags,
                        "relationships": relationships,
                        "summaries": summaries,
                    },
                )

        return chunk_tags, relationships, summaries


def run_step2_extraction(chunks, llm, cfg, ckpt):
    cached = ckpt.load("step2_extraction")
    if cached:
        return cached["chunk_tags"], cached["relationships"], cached["summaries"]

    toc_log.info(">>> STEP 2: TAG AND RELATIONSHIP EXTRACTION")
    toc_log.info(f"  {len(chunks)} chunks, one generation each - this is the slow step")

    partial = ckpt.load("step2_extraction_partial")
    extractor = FineGrainedTagExtractor(llm, cfg)
    chunk_tags, relationships, summaries = extractor.extract_all(chunks, ckpt, partial)

    s = extractor.stats
    toc_log.info(f"  chunks processed : {s['chunks']}  ({s['failed']} failed)")
    toc_log.info(
        f"  raw tags         : {s['tags']}  "
        f"({s['tags'] / max(1, s['chunks']):.1f} per chunk)"
    )
    toc_log.info(f"  relationships    : {s['relationships']}")
    if s["failed"]:
        toc_log.warning(
            f"  {s['failed']} chunks produced no tags and will never be "
            f"assigned to a section"
        )

    ckpt.save(
        "step2_extraction",
        {
            "chunk_tags": chunk_tags,
            "relationships": relationships,
            "summaries": summaries,
        },
    )
    return chunk_tags, relationships, summaries
