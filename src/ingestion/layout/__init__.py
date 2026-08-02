"""ingestion.layout - Module 1.3: PDF/DOCX/PPTX/XLSX -> structured blocks (MinerU)."""

from .mineru import LayoutIngestor, parse_content_list, run_mineru

__all__ = ["LayoutIngestor", "run_mineru", "parse_content_list"]
