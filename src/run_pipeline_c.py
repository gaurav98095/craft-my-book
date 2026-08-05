"""
Pipeline C — Book Writer, run end to end.

Requires Pipeline B to have produced storage/toc.json (run_pipeline_b.py).
Writes one section at a time into output/sections/, with state checkpointed
after every section so a crash at section 90 resumes at section 90.

    python -m src.run_pipeline_c

Start with RUN_LIMIT = 1 below: read the first section and its ledger diff
before letting the full run go unattended.
"""

from .llm import build_llm_client

from .book_writer.setup import BOOK, WriterConfig, writer_log
from .book_writer.constitution import Constitution, DEFAULT_CONSTITUTION
from .book_writer.ledger import BookLedger, load_toc, seed_ledger_from_pipeline_b
from .book_writer.draft_store import DraftStore
from .book_writer.source_memory import SourceMemory
from .book_writer.context_assembler import ContextAssembler
from .book_writer.memory import ClassroomMemory, ConversationMemory, MemoryManager
from .book_writer.agents import (
    WriterAgent,
    ReviewerAgent,
    StudentAgent,
    EditorAgent,
    ArchivistAgent,
)
from .book_writer.continuity_gate import ContinuityGate
from .book_writer.orchestrator import BookOrchestrator
from .book_writer.finishing import run_finishing_passes

# Raise this, or set to None, once you have read the first section and its
# ledger diff and are satisfied the run is behaving.
RUN_LIMIT = 1


def main() -> None:
    # -- the shared model -----------------------------------------------------
    llm = build_llm_client(logger=writer_log)
    writer_log.info(f"Pipeline C model: {llm.model_id}")

    wcfg = WriterConfig()

    # -- Phase 1: Constitution and Book Ledger ---------------------------------
    constitution = Constitution(BOOK.constitution, DEFAULT_CONSTITUTION)
    ledger = BookLedger(BOOK.ledger)
    toc = load_toc()
    seed_stats = seed_ledger_from_pipeline_b(ledger)
    print(f"Phase 1 complete — \"{toc['book_title']}\", ledger seeded: {seed_stats}")

    # -- Phase 2: Draft Store and Source Memory --------------------------------
    drafts = DraftStore(BOOK)
    source = SourceMemory(ledger, neighbourhood=1)
    print(
        f"Phase 2 complete — {len(drafts.written_ids())} sections on disk, "
        f"{len(source.meta)} source chunks"
    )

    # -- Phase 3: Context Assembler and working memory -------------------------
    assembler = ContextAssembler(constitution, ledger, drafts, source, toc)
    classroom = ClassroomMemory(BOOK)
    conversation = ConversationMemory(BOOK, llm, threshold=wcfg.compaction_threshold)
    memory = MemoryManager(
        constitution, ledger, drafts, source, assembler, classroom, conversation
    )
    print("Phase 3 complete — context assembler and working memory ready")

    # -- Phase 4: the five agents -----------------------------------------------
    writer = WriterAgent(llm, wcfg)
    reviewer = ReviewerAgent(llm, wcfg)
    student = StudentAgent(llm, wcfg)
    editor = EditorAgent(llm, wcfg)
    archivist = ArchivistAgent(llm, wcfg)
    print("Phase 4 complete — Writer, Reviewer, Student, Editor, Archivist ready")

    # -- Phase 5: gate, guard, orchestrator --------------------------------------
    gate = ContinuityGate(ledger, drafts, wcfg.repetition_threshold)
    orchestrator = BookOrchestrator(
        toc,
        memory,
        {
            "writer": writer,
            "reviewer": reviewer,
            "student": student,
            "editor": editor,
            "archivist": archivist,
        },
        gate,
        wcfg,
    )
    print(
        f"Phase 5 complete — {len(orchestrator.queue.pending)} pending, "
        f"{len(orchestrator.queue.completed)} already written"
    )

    # -- write the book -----------------------------------------------------------
    run_report = orchestrator.run(limit=RUN_LIMIT)
    print("=" * 72)
    print("RUN REPORT")
    print("=" * 72)
    for key in ("sections_written", "words", "failed", "remaining", "minutes"):
        print(f"  {key:20s}: {run_report[key]}")
    print("\nAgent activity:")
    for agent in (writer, reviewer, student, editor, archivist):
        print(f"  {agent.name:<10} calls={agent.calls:<5} failures={agent.failures}")
    print("\nMemory after the run:")
    for key, value in run_report["memory"].items():
        print(f"  {key:20s}: {value}")

    # -- Phase 6: finishing passes, once sections exist ----------------------------
    if ledger._cache["section_summaries"]:
        final_report = run_finishing_passes(
            toc, ledger, drafts, source, llm, run_contradictions=True
        )
        print("=" * 72)
        print(f"PHASE 6 COMPLETE — \"{final_report['book_title']}\"")
        print("=" * 72)
        print(
            f"Sections     : {final_report['sections_written']} / "
            f"{final_report['sections_planned']}"
        )
        print(
            f"Length       : {final_report['words']:,} words "
            f"≈ {final_report['estimated_pages']} pages"
        )
        print(f"Promises UNKEPT        : {len(final_report['promises']['gaps'])}")
        print(f"Possible contradictions: {len(final_report['contradictions'])}")
        print(f"Rough seams            : {len(final_report['rough_seams'])}")
        print(f"Source coverage        : {final_report['coverage']['coverage_pct']}%")
        print(
            "\nWritten to output/book/: manuscript.md, glossary.md, index.md, "
            "completeness_report.md/.json"
        )
    else:
        print("No sections written yet — raise RUN_LIMIT and re-run.")

    llm.cleanup()


if __name__ == "__main__":
    main()
