"""ingestion.vision - Module 1.4: figure/table/equation understanding.

Model/provider is entirely config-driven (config.INGESTION.vision.vlm) - see
ingestion.vision.describe for details.
"""

from .describe import describe_visuals

__all__ = ["describe_visuals"]
