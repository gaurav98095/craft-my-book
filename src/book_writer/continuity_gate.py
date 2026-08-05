"""Phase 5.1 — The Continuity Gate."""

from typing import Dict, List

from .ledger import BookLedger
from .draft_store import DraftStore


class ContinuityGate:
    """Five deterministic checks on the teaching plan. Zero LLM calls."""

    def __init__(self, ledger: BookLedger, drafts: DraftStore, threshold: float = 0.85):
        self.ledger = ledger
        self.drafts = drafts
        self.REPETITION_THRESHOLD = threshold

    def check_plan(self, section: Dict, plan: Dict) -> Dict:
        issues: List[Dict] = []
        plan_text = " ".join(
            f"{s.get('title', '')} {s.get('topic', '')}" for s in plan.get("steps", [])
        )

        # 1. REPETITION — is this plan too close to something already written?
        for sim in self.drafts.find_similar(
            plan_text, k=3, exclude=section["section_id"]
        ):
            if sim["similarity"] > self.REPETITION_THRESHOLD:
                issues.append(
                    {
                        "type": "repetition",
                        "severity": "high",
                        "message": (
                            f"This plan is {sim['similarity']:.0%} similar to "
                            f"'{sim['title']}' ({sim['section_id']}). Reference it "
                            f"and go deeper, or narrow this section's scope."
                        ),
                    }
                )

        # 2. REDEFINITION — is it planning to define an already-defined term?
        for step in plan.get("steps", []):
            key = self.ledger.resolve_alias(step.get("title", ""))
            if key:
                c = self.ledger._cache["concepts"][key]
                if c["definition"] and c["defined_in"] != section["section_id"]:
                    issues.append(
                        {
                            "type": "redefinition",
                            "severity": "medium",
                            "message": (
                                f"'{c['canonical_name']}' was already defined in "
                                f"{c['defined_in']}. Build on it, do not restate it."
                            ),
                        }
                    )

        # 3. PREREQUISITES — does it lean on something not yet taught?
        missing = self.ledger.undefined_prerequisites(section.get("tags", []))
        if missing:
            issues.append(
                {
                    "type": "prerequisite_gap",
                    "severity": "high",
                    "message": (
                        f"Uses concepts the book has not defined yet: "
                        f"{', '.join(missing)}. Define them briefly here, or the "
                        f"reader is lost."
                    ),
                }
            )

        # 4. STALE SOURCES — is every assigned chunk already fully mined?
        fresh = [
            c for c in section["chunk_ids"] if not self.ledger.chunk_already_used(c)
        ]
        if not fresh and section["chunk_ids"]:
            issues.append(
                {
                    "type": "no_fresh_source",
                    "severity": "medium",
                    "message": (
                        "Every assigned chunk has been used elsewhere. This section "
                        "needs a genuinely new angle or it will read as filler."
                    ),
                }
            )

        # 5. UNKEPT PROMISES — is this the chapter that owed the reader something?
        for p in self.ledger.open_promises_for(section["chapter_id"]):
            if p["text"].lower() not in plan_text.lower():
                issues.append(
                    {
                        "type": "unfulfilled_promise",
                        "severity": "medium",
                        "message": (
                            f"{p['made_in']} promised: \"{p['text']}\" — this chapter "
                            f"is where it was due. Cover it or move the promise."
                        ),
                    }
                )

        return {
            "passed": not any(i["severity"] == "high" for i in issues),
            "issues": issues,
        }
