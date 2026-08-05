"""
Pipeline B — Autonomous Table of Contents Generation.

Reads Pipeline A's chunks and produces storage/toc.json: chapters, sections,
their source chunk ids, and word budgets, via tag extraction, normalization,
clustering into chapters/sections, and curriculum ordering.
"""
