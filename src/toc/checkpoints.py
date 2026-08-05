"""
Step B0.3 — Checkpoints.

Every step writes a checkpoint keyed by name, so a run that dies at step 6
does not repay the LLM bill for steps 1-5. Kept deliberately simple: JSON on
disk, one file per step, under Pipeline A's checkpoint directory.
"""

import re
import json
from pathlib import Path
from typing import Any, Optional

from .setup import toc_log


class StepCheckpoints:
    def __init__(self, directory: Path, enabled: bool = True):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

    def _path(self, name: str) -> Path:
        return self.dir / f"pipelineB_{re.sub(r'[^A-Za-z0-9_]', '_', name)}.json"

    def load(self, name: str) -> Optional[Any]:
        if not self.enabled:
            return None
        path = self._path(name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            toc_log.info(f"  resumed '{name}' from checkpoint")
            return data
        except Exception as exc:
            toc_log.warning(f"  checkpoint '{name}' unreadable ({exc}); recomputing")
            return None

    def save(self, name: str, data: Any) -> None:
        try:
            self._path(name).write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            toc_log.error(f"  could not save checkpoint '{name}': {exc}")
