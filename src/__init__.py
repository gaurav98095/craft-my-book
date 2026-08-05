"""
src — an agentic pipeline that turns raw lecture recordings and
documents into a full-length written book.

Three stages, each runnable on its own:

    ingestion     Pipeline A — raw sources -> a chunked, tagged-ready corpus
    toc           Pipeline B — corpus -> storage/toc.json (chapters, sections)
    book_writer   Pipeline C — toc.json -> a written manuscript

Run them in order:

    python -m src.run_pipeline_a
    python -m src.run_pipeline_b
    python -m src.run_pipeline_c
"""
