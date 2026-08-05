"""
Phase 5.3 — The Orchestrator.

The per-section flow, in one class.

One ordering rule matters above all: everything downstream reads the EDITED
text, not the raw steps. The saved markdown is the book; the Archivist
catalogues what the reader will see; the draft store's get_tail() must
return the real closing line.

The exception is close_session, which reads the RAW session -- deferred
doubts are a property of the lesson, not of the prose.
"""

import re
import json
import time
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional

from .setup import BOOK, WriterConfig, writer_log
from .memory import MemoryManager
from .continuity_gate import ContinuityGate
from .edit_guard import verify_edit
from .work_queue import WorkQueue
from .agents.base import BaseAgent
from .agents.editor import mask_code, unmask_code
from .agents.archivist import ArchivistAgent


class BookOrchestrator:
    def __init__(
        self,
        toc: Dict,
        memory: MemoryManager,
        agents: Dict[str, BaseAgent],
        gate: ContinuityGate,
        cfg: WriterConfig,
    ):
        self.toc = toc
        self.sections = {s["section_id"]: s for s in toc["sections"]}
        self.memory = memory
        self.writer = agents["writer"]
        self.reviewer = agents["reviewer"]
        self.student = agents["student"]
        self.editor = agents["editor"]
        self.archivist = agents["archivist"]
        self.gate = gate
        self.cfg = cfg
        self.queue = WorkQueue.build(toc, cfg.resume)
        self.stats = Counter()

    # ------------------------------------------------------------ planning ---
    def _run_planning_loop(self, section: Dict, context: Dict) -> Dict:
        """Plan, gate, review. The gate has memory; the reviewer has judgement."""
        sid = section["section_id"]
        plan, feedback = None, ""

        for attempt in range(1, self.cfg.max_plan_revisions + 1):
            plan = self.writer.create_teaching_plan(section, context, feedback)
            if not plan or not plan.get("steps"):
                feedback = (
                    "You returned no steps. Return a plan with 3-5 concrete steps."
                )
                self.stats["plan_empty"] += 1
                continue

            gate_result = self.gate.check_plan(section, plan)
            review = self.reviewer.review_plan(
                section, plan, context, self.memory.constitution
            )

            for issue in gate_result["issues"]:
                writer_log.info(
                    f"  [{sid}] gate {issue['severity']}: {issue['message']}"
                )
                self.stats[f"gate_{issue['type']}"] += 1

            if gate_result["passed"] and review.get("approved"):
                self.memory.classroom.set_plan(sid, plan)
                writer_log.info(
                    f"  [{sid}] plan approved on attempt {attempt} "
                    f"({len(plan['steps'])} steps)"
                )
                return plan

            feedback = (
                str(review.get("feedback", ""))
                + "\n\nCONTINUITY ISSUES:\n"
                + "\n".join(
                    f"- [{i['severity']}] {i['message']}" for i in gate_result["issues"]
                )
            )
            self.memory.classroom.set_plan(sid, plan, feedback)
            self.memory.conversation.add(sid, "reviewer", feedback)
            self.stats["plan_rejected"] += 1
            writer_log.warning(f"  [{sid}] plan rejected on attempt {attempt}")

        # Out of revisions: proceed with the last plan rather than losing the
        # section. The gate's issues are already logged and counted.
        writer_log.warning(f"  [{sid}] proceeding with an unapproved plan")
        self.stats["plan_forced"] += 1
        return plan or {
            "steps": [{"title": section["title"], "topic": ""}],
            "teaches": section.get("tags", []),
            "assumes": [],
        }

    # ------------------------------------------------------------ teaching ---
    def _run_teaching_loop(
        self, section: Dict, plan: Dict, context: Dict, prev_tail: str
    ) -> List[Dict]:
        sid = section["section_id"]
        raw_steps: List[Dict] = []
        drafted = ""

        for index, step in enumerate(plan["steps"]):
            prose = self.writer.write_step(
                section,
                plan,
                index,
                context,
                drafted,
                prev_tail,
                conversation=self.memory.conversation.render(sid),
            )
            self.memory.conversation.add(sid, "writer", prose)
            record = {
                "title": step.get("title", f"Step {index + 1}"),
                "prose": prose,
                "clarifications": [],
            }

            # The Student reads the prose as a target reader would.
            for round_no in range(self.cfg.max_doubt_rounds):
                verdict = self.student.evaluate(prose)
                self.memory.conversation.add(sid, "student", json.dumps(verdict))
                if verdict["verdict"] == "UNDERSTOOD":
                    break

                doubt = verdict["doubt"]
                self.stats["doubts_raised"] += 1
                writer_log.info(f"  [{sid}] step {index + 1} doubt: {doubt[:80]}")

                if verdict.get("can_wait"):
                    # A deferred doubt is a promise in disguise; Layer 5
                    # promotes it to the ledger when the session closes.
                    self.memory.classroom.record_doubt(
                        sid,
                        doubt,
                        "deferred to a later section",
                        deferred=True,
                        suggested_chapter=None,
                    )
                    self.stats["doubts_deferred"] += 1
                    break

                answer = self.writer.clarify(section, prose, doubt, context)
                self.memory.conversation.add(sid, "writer", answer)
                record["clarifications"].append({"question": doubt, "answer": answer})
                self.memory.classroom.record_doubt(sid, doubt, answer, deferred=False)
                self.stats["doubts_resolved"] += 1
                # re-read the prose WITH the clarification appended
                prose = f"{prose}\n\n{answer}"

            raw_steps.append(record)
            drafted += (
                "\n\n"
                + record["prose"]
                + "".join("\n\n" + c["answer"] for c in record["clarifications"])
            )
            self.memory.classroom.get_session(sid)["steps_completed"] = index + 1

        return raw_steps

    # -------------------------------------------------------------- editing --
    @staticmethod
    def _concat_steps(raw_steps: List[Dict]) -> str:
        parts = []
        for step in raw_steps:
            parts.append(step["prose"])
            parts.extend(c["answer"] for c in step["clarifications"])
        return "\n\n".join(p for p in parts if p and p.strip())

    def _edit(
        self, section: Dict, raw_steps: List[Dict], plan: Dict, prev_tail: str
    ) -> str:
        """
        Melt the steps into one section, then verify mechanically.

        On a second failure we ship the concatenated raw draft: a slightly
        bumpy section is a much better outcome than a smooth one that lost a
        definition.
        """
        sid = section["section_id"]
        raw_text = self._concat_steps(raw_steps)
        if not raw_text.strip():
            return ""

        # Code blocks never reach the Editor. Each step is masked locally,
        # then its placeholders are renumbered into one global sequence in a
        # SINGLE regex pass. Sequential .replace() calls collide: with offset
        # 1, renaming 0->1 and then 1->2 hits the placeholder just created,
        # block 1 loses its marker, and the guard forces a raw fallback on
        # every section with more than one code block -- silently disabling
        # the Editor exactly where it matters most.
        def _renumber(text: str, offset: int) -> str:
            return re.sub(
                r"\[\[CODE_BLOCK_(\d+)\]\]",
                lambda m: f"[[CODE_BLOCK_{offset + int(m.group(1))}]]",
                text,
            )

        masked_steps = []
        all_blocks: List[str] = []
        for step in raw_steps:
            masked_prose, blocks = mask_code(step["prose"])
            masked_prose = _renumber(masked_prose, len(all_blocks))
            all_blocks.extend(blocks)
            masked_clarifications = []
            for c in step["clarifications"]:
                m, b = mask_code(c["answer"])
                m = _renumber(m, len(all_blocks))
                all_blocks.extend(b)
                masked_clarifications.append({"question": c["question"], "answer": m})
            masked_steps.append(
                {
                    "title": step["title"],
                    "prose": masked_prose,
                    "clarifications": masked_clarifications,
                }
            )

        masked_raw = self._concat_steps(masked_steps)
        taught = plan.get("teaches", []) or section.get("tags", [])
        feedback = ""

        for attempt in range(self.cfg.max_editor_retries + 1):
            result = self.editor.smooth_section(
                section,
                masked_steps,
                self.memory.constitution,
                prev_tail
                + (f"\n\n[PREVIOUS EDIT REJECTED: {feedback}]" if feedback else ""),
            )
            if not result or not str(result.get("content", "")).strip():
                feedback = "you returned no content"
                continue

            issues = verify_edit(masked_raw, result["content"], all_blocks, taught)
            if not issues:
                self.stats["edited"] += 1
                return unmask_code(result["content"], all_blocks)

            writer_log.warning(f"  [{sid}] edit guard: {issues}")
            feedback = "; ".join(issues)
            self.stats["edit_retry"] += 1

        writer_log.warning(f"  [{sid}] editor failed twice — shipping the raw draft")
        self.stats["edit_fallback"] += 1
        return raw_text

    # -------------------------------------------------------- one section ----
    def process_section(self, sid: str) -> Dict[str, Any]:
        section = self.sections[sid]
        started = time.time()
        writer_log.info(
            f"[{sid}] {section['title']}  "
            f"(target {section.get('estimated_word_count', 700)} words)"
        )

        self.memory.classroom.start_session(sid, section)
        prev_id = self.queue.prev_section_id

        # 1. Build the prompt from all memory layers.
        context = self.memory.build_context(section, prev_id)
        prev_tail = self.memory.previous_tail(prev_id, self.cfg.tail_chars)

        # 2. Plan, gate, review.
        plan = self._run_planning_loop(section, context)

        # 3. Draft, step by step, with the Student.
        raw_steps = self._run_teaching_loop(section, plan, context, prev_tail)
        if self.cfg.save_raw_steps:
            (BOOK.raw_steps / f"{sid}.json").write_text(
                json.dumps(raw_steps, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        # 4. Edit into one continuous section.
        final_content = self._edit(section, raw_steps, plan, prev_tail)
        if not final_content.strip():
            raise RuntimeError("section produced no content")

        # 5. Save. This file is the book, and it is written before the
        #    Archivist runs so a cataloguing failure cannot lose the prose.
        self.memory.drafts.write_markdown(sid, section["title"], final_content)

        # 6. Write back into memory. The Archivist reads the EDITED text.
        update = self.archivist.harvest(
            section, final_content, self.memory.ledger.all_open_promises()
        )
        if update is None:
            writer_log.error(f"  [{sid}] archivist failed — recording a degraded entry")
            update = ArchivistAgent.fallback_update(section, final_content)
            self.stats["archivist_failed"] += 1

        summary = update.setdefault("summary", {})
        summary.setdefault("title", section["title"])
        summary["word_count"] = len(final_content.split())

        self.memory.ledger.apply_archivist_update(update, sid)
        # Track which figures this section leaned on, so _select_figures can
        # prefer fresh diagrams next time ("the same diagram is not leaned on
        # twice").
        self.memory.ledger.record_figures_used(
            [f["id"] for f in context.get("figures", [])], sid
        )
        self.memory.drafts.add_section(
            sid,
            section["title"],
            section["chapter_id"],
            final_content,
            summary.get("abstract", ""),
        )
        self.memory.classroom.close_session(sid, self.memory.ledger)
        self.memory.conversation.close(sid)

        elapsed = time.time() - started
        words = len(final_content.split())
        self.stats["sections"] += 1
        self.stats["words"] += words
        writer_log.info(
            f"  [{sid}] done: {words} words in {elapsed / 60:.1f} min "
            f"(ledger v{self.memory.ledger._cache['version']})"
        )
        return {
            "section_id": sid,
            "words": words,
            "seconds": round(elapsed, 1),
            "degraded": bool(update.get("_degraded")),
        }

    # ------------------------------------------------- chapter boundaries ----
    def _maybe_roll_up(self, finished_sid: str) -> None:
        """
        Chapter rollups keep the spine small as the book grows.

        Written once per chapter, at the boundary, by the Archivist -- so
        that by section 120 the other eleven chapters cost a paragraph each
        rather than 119 abstracts competing for the model's attention.
        """
        chapter_id = self.sections[finished_sid]["chapter_id"]
        siblings = [
            s["section_id"]
            for s in self.toc["sections"]
            if s["chapter_id"] == chapter_id
        ]
        written = set(self.memory.ledger._cache["section_summaries"])
        if not set(siblings) <= written:
            return
        if chapter_id in self.memory.ledger._cache["chapter_rollups"]:
            return

        abstracts = "\n".join(
            f"- {sid}: {self.memory.ledger._cache['section_summaries'][sid].get('abstract', '')}"
            for sid in siblings
        )
        title = next(
            c["title"] for c in self.toc["chapters"] if c["chapter_id"] == chapter_id
        )
        try:
            rollup = self.archivist._execute_step(
                f'CHAPTER {chapter_id}: "{title}"\n\nSECTION ABSTRACTS:\n{abstracts}\n\n'
                f"Write ONE paragraph (60-80 words) summarising what this chapter built "
                f"and what the reader can do at the end of it. Prose only.",
                "You are the Archivist. You summarise finished chapters factually.",
                # See the reasoning-budget note in book_writer/setup.py.
                max_tokens=1_500,
            )
            if rollup:
                self.memory.ledger.add_chapter_rollup(chapter_id, rollup.strip())
                writer_log.info(f"  chapter rollup written for {chapter_id}")
        except Exception as exc:
            writer_log.warning(f"  rollup for {chapter_id} failed: {exc}")

    # ----------------------------------------------------------------- run ---
    def run(self, limit: Optional[int] = None) -> Dict[str, Any]:
        writer_log.info("=" * 70)
        writer_log.info(
            f"WRITING: \"{self.toc['book_title']}\" — "
            f"{len(self.queue.pending)} sections to go"
        )
        writer_log.info("=" * 70)
        started = time.time()
        done = 0

        while self.queue.pending and (limit is None or done < limit):
            sid = self.queue.pending[0]
            try:
                self.process_section(sid)
                self.queue.pending.pop(0)
                self.queue.completed.append(sid)
                self.queue.prev_section_id = sid
                self._maybe_roll_up(sid)
            except Exception as exc:
                writer_log.error(f"[{sid}] FAILED: {exc}")
                writer_log.debug(traceback.format_exc())
                self.queue.pending.pop(0)
                self.queue.failed.append({"section_id": sid, "error": str(exc)})
                self.stats["failed"] += 1
                if self.cfg.stop_on_error:
                    self.queue.save()
                    raise
            # Checkpoint after every section: a crash at 90 resumes at 90.
            self.queue.save()
            done += 1

        elapsed = time.time() - started
        return {
            "sections_written": self.stats["sections"],
            "words": self.stats["words"],
            "failed": self.stats["failed"],
            "remaining": len(self.queue.pending),
            "minutes": round(elapsed / 60, 1),
            "stats": dict(self.stats),
            "memory": self.memory.snapshot(),
        }
