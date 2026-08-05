"""Steps 4-7 — Clustering into chapters and sections: theme discovery, tag
assignment, section formation, and curriculum ordering."""

import math
from collections import Counter, defaultdict
from typing import Dict, List

from tqdm.auto import tqdm

from .setup import BookSizeConfig, PipelineBConfig, toc_log
from ..llm import LLMClient

THEME_SCHEMA = {
    "themes": [
        {
            "id": "string",
            "title": "string",
            "description": "string",
            "example_concepts": ["string"],
            "difficulty_level": "foundational|intermediate|advanced",
        }
    ],
    "reasoning": "string",
}

ASSIGNMENT_SCHEMA = {
    "assignments": {"concept": "theme_id"},
    "uncategorized": ["string"],
    "reasoning": "string",
}


class ThemeDiscovery:
    """Phase 1: identify the chapter-sized themes. Prompt preserved verbatim."""

    def __init__(self, llm: LLMClient, cfg: PipelineBConfig):
        self.llm = llm
        self.cfg = cfg
        self.system_prompt = f"""You are an expert curriculum designer analyzing concepts to identify major learning themes.

YOUR TASK: Given a sample of concepts from a technical curriculum, identify the {cfg.size.target_chapters_min}-{cfg.size.target_chapters_max} major themes/chapters that would organize ALL similar concepts.

PRINCIPLES FOR GOOD THEMES:
1. Themes should be MUTUALLY EXCLUSIVE - a concept should clearly belong to one theme
2. Themes should be COLLECTIVELY EXHAUSTIVE - all concepts should fit somewhere
3. Themes should follow PEDAGOGICAL PROGRESSION - foundational to advanced
4. Themes should be at the right GRANULARITY - not too broad, not too narrow

EXAMPLE FOR MACHINE LEARNING CURRICULUM:
Good themes:
- "Mathematical Foundations" (linear algebra, calculus, probability)
- "Optimization & Training" (gradient descent, learning rate, convergence)
- "Neural Network Basics" (perceptrons, activation functions, backpropagation)
- "Attention & Transformers" (self-attention, positional encoding, BERT)
- "Regularization & Generalization" (dropout, batch norm, overfitting)

Bad themes:
- "Deep Learning" (too broad - should be split)
- "Advanced Topics" (too vague)
- "Part 1" (not descriptive)

Identify {cfg.size.target_chapters_min}-{cfg.size.target_chapters_max} themes that would create a well-structured curriculum."""

    def discover(self, chunk_tags: Dict[str, List[str]]) -> List[Dict]:
        counts = Counter(t for tags in chunk_tags.values() for t in tags)
        sample = [t for t, _ in counts.most_common(self.cfg.theme_sample_size)]

        user = (
            f"CONCEPTS FROM THE CORPUS ({len(counts)} total, "
            f"{len(sample)} most frequent shown):\n{', '.join(sample)}\n\n"
            f"Identify the major themes."
        )

        reply = self.llm.generate_structured(
            self.system_prompt,
            user,
            THEME_SCHEMA,
            max_tokens=self.cfg.max_tokens_structure,
            temperature=0.3,
            max_attempts=self.cfg.structured_max_attempts,
        )

        themes = (reply or {}).get("themes", [])
        cleaned = []
        for i, theme in enumerate(themes, 1):
            cleaned.append(
                {
                    "id": str(theme.get("id") or f"theme_{i:02d}"),
                    "title": str(theme.get("title", f"Theme {i}")).strip(),
                    "description": str(theme.get("description", "")).strip(),
                    "difficulty_level": theme.get("difficulty_level", "intermediate"),
                }
            )
        if not cleaned:
            raise RuntimeError("Theme discovery returned nothing - cannot build a TOC")
        return cleaned


class TagAssigner:
    """Phase 2: every concept into exactly one chapter. Prompt preserved verbatim."""

    def __init__(self, llm: LLMClient, cfg: PipelineBConfig):
        self.llm = llm
        self.cfg = cfg
        self.system_prompt = """You are an expert curriculum designer assigning concepts to chapters.

YOUR TASK: Assign each concept to the SINGLE most appropriate theme/chapter.

RULES:
1. Every concept MUST be assigned to exactly ONE theme
2. Choose the MOST SPECIFIC theme that fits
3. If a concept could fit multiple themes, choose the one where it's MOST CENTRAL
4. Use "UNCATEGORIZED" only for concepts that truly don't fit anywhere

Assign EVERY concept to a theme. Minimize uncategorized."""

    def assign(
        self, themes: List[Dict], chunk_tags: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        counts = Counter(t for tags in chunk_tags.values() for t in tags)
        all_tags = [t for t, _ in counts.most_common()]

        theme_block = "\n".join(
            f"  {t['id']}: \"{t['title']}\" - {t['description']}" for t in themes
        )
        valid_ids = {t["id"] for t in themes}

        assignments: Dict[str, str] = {}
        size = self.cfg.assignment_batch_size
        batches = [all_tags[i : i + size] for i in range(0, len(all_tags), size)]

        for batch in tqdm(batches, desc="Assigning tags to chapters"):
            user = (
                f"THEMES:\n{theme_block}\n\n"
                f"CONCEPTS TO ASSIGN ({len(batch)}):\n{', '.join(batch)}\n\n"
                f"Assign every concept above to one theme id."
            )
            reply = self.llm.generate_structured(
                self.system_prompt,
                user,
                ASSIGNMENT_SCHEMA,
                max_tokens=self.cfg.max_tokens_structure,
                temperature=0.2,
                max_attempts=self.cfg.structured_max_attempts,
            )

            for tag, theme_id in (reply or {}).get("assignments", {}).items():
                tag = str(tag).strip().lower()
                if tag in counts and str(theme_id) in valid_ids:
                    assignments[tag] = str(theme_id)

        # An unassigned tag is not a rounding error -- it is source material
        # that can never reach a section. Park it in the nearest chapter by
        # co-occurrence rather than dropping it, and report the count.
        unassigned = [t for t in all_tags if t not in assignments]
        if unassigned:
            toc_log.warning(
                f"  {len(unassigned)} concepts unassigned by the model; "
                f"placing them by co-occurrence"
            )
            cooccurrence = defaultdict(Counter)
            for tags in chunk_tags.values():
                for tag in tags:
                    if tag in unassigned:
                        for other in tags:
                            if other in assignments:
                                cooccurrence[tag][assignments[other]] += 1
            fallback = themes[0]["id"]
            for tag in unassigned:
                best = cooccurrence[tag].most_common(1)
                assignments[tag] = best[0][0] if best else fallback

        by_chapter: Dict[str, List[str]] = defaultdict(list)
        for tag, theme_id in assignments.items():
            by_chapter[theme_id].append(tag)
        return dict(by_chapter)


SECTION_SCHEMA = {
    "sections": [{"title": "string", "concepts": ["string"]}],
    "reasoning": "string",
}


def allocate_section_quota(
    tags_per_chapter: Dict[str, int], size: BookSizeConfig
) -> Dict[str, int]:
    """
    Split the book's section budget across chapters, in proportion to how much
    vocabulary each one received.

    This is where the design's page arithmetic actually lands. Without it, the
    section count is whatever the model felt like, and the book comes out at a
    fifth of the requested length.
    """
    total_tags = sum(tags_per_chapter.values()) or 1
    target = size.target_sections
    n_chapters = len(tags_per_chapter) or 1

    # The configured floor and ceiling are sanity caps, not hard truths. If the
    # chapter count and the section target disagree with them, the caps must
    # give way -- otherwise the loop below runs out of chapters to adjust and
    # returns a total that quietly misses the target. With 4 chapters, a
    # target of 175 and a ceiling of 24, that silently produced 96.
    # The ceiling scales with the book: capping at a fixed 24 when the mean is
    # already 17.5 flattens the distribution so hard that a chapter with 300
    # concepts and one with 120 get the same number of sections. Allow roughly
    # twice the mean, which keeps the cap meaningful without erasing the shape.
    ceiling = max(size.max_sections_per_chapter, math.ceil(2.0 * target / n_chapters))
    floor = min(size.min_sections_per_chapter, max(1, target // n_chapters))

    quota = {}
    for chapter_id, n_tags in tags_per_chapter.items():
        share = target * n_tags / total_tags
        quota[chapter_id] = int(max(floor, min(ceiling, round(share))))

    # Rebalance towards the target: rounding and clamping push the total off, so
    # give or take sections from the chapters with the most vocabulary first.
    order = sorted(quota, key=lambda c: -tags_per_chapter[c])
    guard = 0
    while sum(quota.values()) != target and guard < 100_000:
        guard += 1
        if sum(quota.values()) > target:
            movable = [c for c in reversed(order) if quota[c] > floor]
            if not movable:
                break
            quota[movable[0]] -= 1
        else:
            movable = [c for c in order if quota[c] < ceiling]
            if not movable:
                break
            quota[movable[0]] += 1

    # If the caps still could not be reconciled with the target, say so. A book
    # that comes out a third of its requested length is not a rounding error.
    achieved = sum(quota.values())
    if achieved != target:
        toc_log.warning(
            f"  section budget: asked for {target}, allocated {achieved} "
            f"across {n_chapters} chapters (floor={floor}, ceiling={ceiling}) "
            f"- adjust target_pages or the per-chapter limits"
        )

    return quota


class SectionFormer:
    """
    Phase 3: break each chapter into sections.

    Two levels only: chapter -> section. Pipeline C reads `chapter_id` and
    `section_id` and there is no third level anywhere in the design.
    """

    def __init__(self, llm: LLMClient, cfg: PipelineBConfig):
        self.llm = llm
        self.cfg = cfg
        self.system_prompt = """You are an expert curriculum designer organizing concepts into sections.

YOUR TASK: Given the concepts belonging to a chapter, organize them into a specific number of sections.

PRINCIPLES:
1. Group related concepts together
2. Order sections from foundational to advanced within the chapter
3. Every concept must appear in exactly ONE section
4. Every section must contain at least one concept
5. Section titles should be specific and teachable, not "Introduction" or "Advanced Topics"

A section becomes roughly 700 words of a book -- one focused idea, taught once.
If you are asked for 12 sections, return exactly 12."""

    def form(self, chapter: Dict, tags: List[str], quota: int) -> List[Dict]:
        user = f"""CHAPTER: "{chapter['title']}"
DESCRIPTION: {chapter.get('description', 'N/A')}

CONCEPTS TO ORGANIZE ({len(tags)}):
{', '.join(sorted(tags))}

Create EXACTLY {quota} sections, ordered foundational to advanced.
Every concept above must appear in exactly one section."""

        reply = self.llm.generate_structured(
            self.system_prompt,
            user,
            SECTION_SCHEMA,
            max_tokens=self.cfg.max_tokens_structure,
            temperature=0.3,
            max_attempts=self.cfg.structured_max_attempts,
        )

        sections = []
        assigned = set()
        for i, section in enumerate((reply or {}).get("sections", []), 1):
            concepts = [
                str(c).strip().lower()
                for c in section.get("concepts", [])
                if str(c).strip().lower() in set(tags)
            ]
            concepts = [c for c in concepts if c not in assigned]
            assigned.update(concepts)
            title = str(section.get("title", "")).strip() or f"Section {i}"
            if concepts:
                sections.append({"title": title, "tags": concepts})

        # Concepts the model forgot go to the section they best match by
        # co-membership; if there is no section at all, make one.
        leftover = [t for t in tags if t not in assigned]
        if leftover:
            if not sections:
                sections.append({"title": chapter["title"], "tags": []})
            toc_log.debug(f"    {len(leftover)} concepts unplaced; appending")
            for tag in leftover:
                sections[-1]["tags"].append(tag)

        if len(sections) != quota:
            toc_log.warning(
                f"  '{chapter['title']}': asked for {quota} sections, "
                f"got {len(sections)}"
            )
        return sections


ORDER_SCHEMA = {"ordered_ids": ["string"], "reasoning": "string"}


class CurriculumOrderer:
    """
    Phase 4: order chapters, and the sections inside each chapter.

    Pure LLM. The model sees each unit's title and a sample of its concepts and
    returns them sequenced foundational-to-advanced. Its domain knowledge is the
    whole mechanism here -- nothing recomputes or second-guesses the sequence.

    The code's only job afterwards is list hygiene: models occasionally drop an
    id or hallucinate one, and a missing chapter would silently vanish from the
    book. So the returned list is reconciled against the input.
    """

    def __init__(self, llm: LLMClient, cfg: PipelineBConfig):
        self.llm = llm
        self.cfg = cfg
        self.notes: List[str] = []
        self.system_prompt = """You are an expert curriculum designer ordering content for optimal learning.

YOUR TASK: Order the items given from foundational to advanced.

PRINCIPLES:
1. Prerequisites come before dependent topics
2. Foundational concepts before advanced applications
3. Theory before implementation
4. General before specific
5. Build complexity gradually

Order by learning progression, not alphabetically or by size.
Return every id you were given, exactly once."""

    def order(self, units: List[Dict], label: str) -> List[Dict]:
        """Return `units` in the order the model chose."""
        if len(units) <= 1:
            return units

        listing = "\n".join(
            f"- {u['id']}: \"{u['title']}\"\n  concepts: "
            f"{', '.join(u['tags'][:10])}"
            for u in units
        )
        user = (
            f"Order these {len(units)} {label} for optimal learning progression:\n\n"
            f"{listing}\n\nReturn the ids in pedagogically correct order."
        )

        reply = self.llm.generate_structured(
            self.system_prompt, user, ORDER_SCHEMA, max_tokens=1_500, temperature=0.2,
            max_attempts=self.cfg.structured_max_attempts,
        )

        proposed = [str(i) for i in (reply or {}).get("ordered_ids", [])]
        known = {u["id"] for u in units}

        # -- list hygiene, not a second opinion ------------------------------
        invented = [p for p in proposed if p not in known]
        proposed = [p for p in proposed if p in known]
        seen, deduped = set(), []
        for p in proposed:  # a repeated id would duplicate a chapter
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        dropped = [u["id"] for u in units if u["id"] not in seen]
        final = deduped + dropped  # anything forgotten keeps its old place

        if invented or dropped:
            note = (
                f"{label}: model returned {len(invented)} unknown id(s) and omitted "
                f"{len(dropped)}; reconciled against the input"
            )
            toc_log.warning(f"  {note}")
            self.notes.append(note)
        if not reply:
            note = f"{label}: ordering call failed, keeping the original order"
            toc_log.warning(f"  {note}")
            self.notes.append(note)

        by_id = {u["id"]: u for u in units}
        return [by_id[i] for i in final]
