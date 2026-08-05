"""Step 3 — Normalization + the artifacts that seed the Book Ledger."""

import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from tqdm.auto import tqdm

from ..ingestion.setup import PATHS
from .setup import PipelineBConfig, toc_log
from .llm import BookLLM
from .checkpoints import StepCheckpoints

NORMALIZATION_SCHEMA = {
    "groups": [
        {
            "canonical": "string - the preferred form",
            "variants": ["string - every raw tag that means this"],
        }
    ]
}


class TagNormalizer:
    """Collapses raw tags into a canonical vocabulary, keeping every alias."""

    def __init__(self, llm: BookLLM, cfg: PipelineBConfig):
        self.llm = llm
        self.cfg = cfg
        self.system_prompt = """You are an expert knowledge engineer building a controlled vocabulary.

YOUR TASK: Group tags that mean the SAME concept under one canonical name.

RULES:
1. Group only true synonyms and spelling/casing variants of the SAME concept
2. Do NOT group related-but-different concepts
   - "self-attention" and "multi-head attention" are DIFFERENT - do not merge
   - "relu" and "sigmoid" are DIFFERENT - do not merge
   - "ReLU", "relu activation", "rectified linear unit" are the SAME - merge these
3. The canonical name should be the clearest, most standard form, lowercase
4. Every input tag must appear in exactly one group
5. A tag with no synonyms forms a group of one

Be conservative. Wrongly merging two concepts is far worse than leaving them separate."""

    def collect_unique(self, chunk_tags: Dict[str, List[str]]) -> List[Tuple[str, int]]:
        counts = Counter(tag for tags in chunk_tags.values() for tag in tags)
        return counts.most_common()

    def normalize(
        self, ranked: List[Tuple[str, int]]
    ) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """
        Returns (tag_to_canonical, canonical_to_variants).

        Tags are batched most-frequent-first so the common vocabulary is settled
        before the long tail arrives, and later batches are shown the canonicals
        already chosen so they can join an existing group instead of starting a
        near-duplicate one.
        """
        tag_to_canonical: Dict[str, str] = {}
        canonical_to_variants: Dict[str, List[str]] = defaultdict(list)

        size = self.cfg.normalization_batch_size
        batches = [ranked[i : i + size] for i in range(0, len(ranked), size)]

        for batch in tqdm(batches, desc="Normalizing tags"):
            known = sorted(canonical_to_variants.keys())[:120]
            user = "TAGS TO GROUP (with corpus frequency):\n"
            user += "\n".join(f"  {tag}  ({count})" for tag, count in batch)
            if known:
                user += (
                    "\n\nCANONICAL NAMES ALREADY CHOSEN — reuse one of these if a "
                    "tag below means the same thing:\n" + ", ".join(known)
                )
            user += "\n\nGroup every tag above."

            reply = self.llm.generate_structured(
                self.system_prompt,
                user,
                NORMALIZATION_SCHEMA,
                max_tokens=self.cfg.max_tokens_normalization,
                temperature=0.0,
            )

            handled = set()
            if reply:
                for group in reply.get("groups", []):
                    canonical = str(group.get("canonical", "")).strip().lower()
                    if not canonical:
                        continue
                    variants = [
                        str(v).strip().lower()
                        for v in group.get("variants", [])
                        if str(v).strip()
                    ]
                    for variant in variants:
                        tag_to_canonical[variant] = canonical
                        if (
                            variant != canonical
                            and variant not in canonical_to_variants[canonical]
                        ):
                            canonical_to_variants[canonical].append(variant)
                        handled.add(variant)
                    canonical_to_variants[canonical]  # touch, so singletons exist

            # Any tag the model skipped becomes its own canonical. Silently
            # losing a tag here would silently lose whatever it points at.
            for tag, _ in batch:
                if tag not in handled:
                    tag_to_canonical[tag] = tag
                    canonical_to_variants[tag]

        return tag_to_canonical, dict(canonical_to_variants)

    @staticmethod
    def apply_to_chunks(
        chunk_tags: Dict[str, List[str]], mapping: Dict[str, str]
    ) -> Dict[str, List[str]]:
        return {
            cid: list(dict.fromkeys(mapping.get(t, t) for t in tags))
            for cid, tags in chunk_tags.items()
        }

    @staticmethod
    def apply_to_relationships(
        relationships: List[Dict], mapping: Dict[str, str]
    ) -> List[Dict]:
        """
        Canonicalise both endpoints, drop self-loops, and merge duplicates while
        counting support.

        Support records how many chunks independently stated the same
        dependency -- useful when reading the ledger, and a cheap signal for
        telling a well-attested prerequisite from a one-off mention.
        """
        merged: Dict[Tuple[str, str, str], Dict] = {}
        for rel in relationships:
            src = mapping.get(rel["from"], rel["from"])
            dst = mapping.get(rel["to"], rel["to"])
            if src == dst:
                continue  # became a self-loop after merging
            key = (src, dst, rel["type"])
            entry = merged.setdefault(
                key,
                {
                    "from": src,
                    "to": dst,
                    "type": rel["type"],
                    "support": 0,
                    "chunk_ids": [],
                },
            )
            entry["support"] += 1
            if len(entry["chunk_ids"]) < 5:
                entry["chunk_ids"].append(rel.get("chunk_id"))
        return list(merged.values())


def run_step3_normalization(chunk_tags, relationships, llm, cfg, ckpt: StepCheckpoints):
    """
    Normalize, then WRITE THE THREE ARTIFACTS. The writing is the point.
    """
    cached = ckpt.load("step3_normalization")
    if cached:
        normalized_chunk_tags = cached["normalized_chunk_tags"]
        normalized_relationships = cached["normalized_relationships"]
        canonical_to_variants = cached["canonical_to_variants"]
    else:
        toc_log.info(">>> STEP 3: TAG NORMALIZATION")
        normalizer = TagNormalizer(llm, cfg)

        ranked = normalizer.collect_unique(chunk_tags)
        toc_log.info(f"  unique raw tags: {len(ranked)}")

        tag_to_canonical, canonical_to_variants = normalizer.normalize(ranked)
        normalized_chunk_tags = normalizer.apply_to_chunks(chunk_tags, tag_to_canonical)
        normalized_relationships = normalizer.apply_to_relationships(
            relationships, tag_to_canonical
        )

        toc_log.info(
            f"  canonical concepts: {len(canonical_to_variants)}  "
            f"(from {len(ranked)} raw tags)"
        )
        toc_log.info(
            f"  relationships: {len(relationships)} → "
            f"{len(normalized_relationships)} after merging"
        )

        ckpt.save(
            "step3_normalization",
            {
                "normalized_chunk_tags": normalized_chunk_tags,
                "normalized_relationships": normalized_relationships,
                "canonical_to_variants": canonical_to_variants,
            },
        )

    # ---- the three artifacts §7 depends on ---------------------------------
    # 1. normalized_tags.json -> BookLedger aliases (drift detection, day one)
    PATHS.normalized_tags.write_text(
        json.dumps(canonical_to_variants, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 2. tag_relationships.json -> BookLedger prerequisites (failure #5)
    #    seed_from_toc_pipeline reads `tag_relationships.get(canonical, [])`,
    #    so the shape is {concept: [things it requires]}.
    prerequisites: Dict[str, List[str]] = defaultdict(list)
    for rel in normalized_relationships:
        if rel["type"] == "requires" and rel["to"] not in prerequisites[rel["from"]]:
            prerequisites[rel["from"]].append(rel["to"])
    PATHS.tag_relationships.write_text(
        json.dumps(dict(prerequisites), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 3. chunk_tags.json -> Layer 3's retrieval index
    PATHS.chunk_tags.write_text(
        json.dumps(normalized_chunk_tags, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    toc_log.info(
        f"  wrote {PATHS.normalized_tags.name}: "
        f"{len(canonical_to_variants)} concepts, "
        f"{sum(len(v) for v in canonical_to_variants.values())} aliases"
    )
    toc_log.info(
        f"  wrote {PATHS.tag_relationships.name}: "
        f"{len(prerequisites)} concepts with prerequisites"
    )
    toc_log.info(
        f"  wrote {PATHS.chunk_tags.name}: {len(normalized_chunk_tags)} chunks"
    )

    return normalized_chunk_tags, normalized_relationships, canonical_to_variants
