"""Stage 2 — Documents to Structured JSON (MinerU / Docling)."""

import json
import time
import threading
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from .setup import (
    PATHS,
    make_logger,
    doc_slug,
    source_type_for,
    SOURCE_TYPE_BY_EXT,
    DOCUMENT_EXTENSIONS,
)

stage2_log = make_logger("stage2.parse", "stage2_parsing.log")


@dataclass
class Stage2Config:
    """Configuration for Stage 2 - Layout parsing."""

    parser_type: str = "mineru"  # "mineru" | "docling"
    parse_method: str = "auto"  # "auto" | "txt" | "ocr"

    # Which files this stage handles. Media files belong to Stage 1, so they
    # are deliberately excluded here rather than being silently "unsupported".
    supported_extensions: List[str] = field(
        default_factory=lambda: sorted(DOCUMENT_EXTENSIONS)
    )

    # MinerU runs models on the GPU. More than one or two workers will fight
    # over VRAM and be slower than sequential, not faster.
    max_workers: int = 2

    # Error handling -- the retry now keys on the RESULT, not on an exception
    max_retries: int = 3
    retry_delay: int = 5
    continue_on_error: bool = True

    # Resume
    skip_existing: bool = True
    force_reparse: bool = False

    validate_output: bool = True


def discover_documents(cfg: Stage2Config) -> Tuple[List[Path], List[Path]]:
    """Split raw_sources into (documents for this stage, files handled elsewhere)."""
    if not PATHS.raw_sources.exists():
        raise FileNotFoundError(f"Put your sources in {PATHS.raw_sources.resolve()}")

    supported = {e.lower() for e in cfg.supported_extensions}
    documents, other = [], []

    for p in sorted(PATHS.raw_sources.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        (documents if p.suffix.lower() in supported else other).append(p)

    stage2_log.info(f"Found {len(documents)} documents, {len(other)} other files")
    if other:
        by_ext: Dict[str, int] = {}
        for p in other:
            by_ext[p.suffix.lower()] = by_ext.get(p.suffix.lower(), 0) + 1
        for ext, n in sorted(by_ext.items(), key=lambda kv: -kv[1]):
            kind = SOURCE_TYPE_BY_EXT.get(ext, "unsupported")
            stage2_log.info(f"  {ext or '(no ext)'}: {n} file(s) -> {kind}")

    return documents, other


# --- parser construction -----------------------------------------------------
# One parser per thread. `threading.local()` gives each worker its own slot in
# a shared object, so no locking is needed and no state is shared.
_parser_local = threading.local()


def get_parser(cfg: Stage2Config):
    """Return this thread's parser, constructing it on first use."""
    existing = getattr(_parser_local, "parser", None)
    if existing is not None:
        return existing

    from raganything.parser import MineruParser, DoclingParser

    kind = cfg.parser_type.lower()
    if kind == "mineru":
        parser = MineruParser()
    elif kind == "docling":
        parser = DoclingParser()
    else:
        raise ValueError(f"Unsupported parser type: {cfg.parser_type}")

    # check_installation() is advisory: some builds report False and still work.
    check = getattr(parser, "check_installation", None)
    if callable(check):
        try:
            if not check():
                stage2_log.warning(
                    f"{kind}: installation check returned False - proceeding anyway"
                )
        except Exception as exc:
            stage2_log.warning(
                f"{kind}: installation check raised ({exc}) - proceeding anyway"
            )

    _parser_local.parser = parser
    return parser


def _normalise_parser_return(raw: Any) -> List[Dict[str, Any]]:
    """
    Accept every shape `parse_document` has returned across versions.

    list                      -> the content list itself
    (content_list, markdown)  -> element 0
    {"content_list": [...]}   -> that key
    """
    if isinstance(raw, tuple) and raw:
        raw = raw[0]
    if isinstance(raw, dict):
        raw = raw.get("content_list", raw.get("content", []))
    if not isinstance(raw, list):
        raise TypeError(f"Parser returned an unusable type: {type(raw).__name__}")
    return raw


def _resolve_image_paths(content_list: List[Dict[str, Any]], parse_dir: Path) -> int:
    """
    Turn every relative `img_path` into a verified absolute path.

    This is the fix for the silent-fallback bug. Parsers emit paths relative to
    their own output directory, and different versions nest them differently
    (`./images/x.jpg`, `auto/images/x.jpg`, ...). We try the likely locations,
    then fall back to a recursive search by filename.

    Items whose crop genuinely cannot be found are marked `img_missing` so
    Stage 3 can report them as a number instead of silently degrading.
    """
    resolved = 0
    for item in content_list:
        rel = item.get("img_path")
        if not rel:
            continue

        candidate = Path(rel)
        if candidate.is_absolute() and candidate.exists():
            resolved += 1
            continue

        found = None
        for base in (
            parse_dir,
            parse_dir / "auto",
            parse_dir / "images",
            parse_dir / "auto" / "images",
        ):
            trial = base / rel
            if trial.exists():
                found = trial
                break

        if found is None:
            matches = list(parse_dir.rglob(Path(rel).name))
            found = matches[0] if matches else None

        if found is not None:
            item["img_path"] = str(found.resolve())
            resolved += 1
        else:
            item["img_missing"] = True

    return resolved


def parse_single_document(file_path: Path, cfg: Stage2Config) -> Dict[str, Any]:
    """
    Parse one document into structured JSON, keeping its image crops on disk.

    Never raises: returns a result dict whose "status" the caller inspects.
    That is a deliberate contract, and the retry loop below honours it.
    """
    t0 = time.time()
    parse_dir = PATHS.parsed / doc_slug(file_path.name)
    parse_dir.mkdir(parents=True, exist_ok=True)

    try:
        parser = get_parser(cfg)

        raw = parser.parse_document(
            file_path=str(file_path),
            output_dir=str(parse_dir),  # <-- the fix: crops land here and stay
            method=cfg.parse_method,
        )
        content_list = _normalise_parser_return(raw)
        images_resolved = _resolve_image_paths(content_list, parse_dir)

        content_types: Dict[str, int] = {}
        for item in content_list:
            t = item.get("type", "unknown")
            content_types[t] = content_types.get(t, 0) + 1

        elapsed = time.time() - t0
        result = {
            "status": "success",
            "file_path": str(file_path),
            "filename": file_path.name,
            "doc_slug": doc_slug(file_path.name),
            "source_type": source_type_for(file_path),  # <-- provenance
            "parse_dir": str(parse_dir),
            "content_list": content_list,
            "metadata": {
                "total_items": len(content_list),
                "content_types": content_types,
                "images_resolved": images_resolved,
                "images_missing": sum(1 for i in content_list if i.get("img_missing")),
                "parse_time": round(elapsed, 2),
                "parse_method": cfg.parse_method,
                "parser_type": cfg.parser_type,
                "parsed_at": datetime.now().isoformat(timespec="seconds"),
                "file_size": file_path.stat().st_size,
            },
        }

        stage2_log.info(
            f"  {file_path.name}: {len(content_list)} items in {elapsed:.1f}s "
            f"| {content_types}"
        )
        if result["metadata"]["images_missing"]:
            stage2_log.warning(
                f"  {file_path.name}: "
                f"{result['metadata']['images_missing']} image crops not found"
            )
        return result

    except Exception as exc:
        stage2_log.error(f"  {file_path.name}: {type(exc).__name__}: {exc}")
        stage2_log.debug(traceback.format_exc())
        return {
            "status": "failed",
            "file_path": str(file_path),
            "filename": file_path.name,
            "doc_slug": doc_slug(file_path.name),
            "source_type": source_type_for(file_path),
            "content_list": [],
            "metadata": {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "parse_time": round(time.time() - t0, 2),
            },
        }


def validate_parsed_output(result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Structural sanity check before we trust a parse."""
    for key in ("status", "filename", "content_list", "metadata"):
        if key not in result:
            return False, f"missing key: {key}"

    if not isinstance(result["content_list"], list):
        return False, "content_list is not a list"

    if result["status"] == "success" and not result["content_list"]:
        return False, "parse succeeded but produced zero content items"

    for i, item in enumerate(result["content_list"]):
        if not isinstance(item, dict):
            return False, f"item {i} is not a dict"
        if "type" not in item:
            return False, f"item {i} has no 'type'"

    return True, None


def parsed_json_path(file_path: Path) -> Path:
    return PATHS.parsed / f"{doc_slug(file_path.name)}.json"


def process_document_with_retry(file_path: Path, cfg: Stage2Config) -> Dict[str, Any]:
    """
    Parse one document, retrying on failure.

    THE FIX: `parse_single_document` reports failure by RETURNING a dict, not by
    raising. Wrapping it in try/except therefore never sees a failure, so
    `max_retries=3` would be decorative. We branch on the status instead.
    """
    out_path = parsed_json_path(file_path)
    last_error = "unknown"

    for attempt in range(1, cfg.max_retries + 1):
        result = parse_single_document(file_path, cfg)

        if result["status"] == "success" and cfg.validate_output:
            ok, msg = validate_parsed_output(result)
            if not ok:
                result["status"] = "failed"
                result["metadata"]["validation_error"] = msg
                stage2_log.error(f"  {file_path.name}: validation failed - {msg}")

        if result["status"] == "success":
            try:
                out_path.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                result["output_path"] = str(out_path)
                result["metadata"]["attempts"] = attempt
                return result
            except Exception as exc:
                last_error = f"could not save result: {exc}"
                stage2_log.error(f"  {file_path.name}: {last_error}")
        else:
            last_error = result["metadata"].get("error", "parse failed")

        if attempt < cfg.max_retries:
            stage2_log.warning(
                f"  {file_path.name}: attempt {attempt}/{cfg.max_retries} "
                f"failed ({last_error}); retrying in {cfg.retry_delay}s"
            )
            time.sleep(cfg.retry_delay)

    stage2_log.error(f"  {file_path.name}: all {cfg.max_retries} attempts failed")
    return {
        "status": "failed",
        "filename": file_path.name,
        "file_path": str(file_path),
        "content_list": [],
        "metadata": {"error": last_error, "attempts": cfg.max_retries},
    }


@dataclass
class Stage2Checkpoint:
    """
    Resumable state.

    Stored as sorted lists (JSON has no set type) but manipulated as sets, so a
    resumed run cannot append the same filename a second time.
    """

    processed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    started_at: str = ""
    updated_at: str = ""

    def merge(self, bucket: str, name: str) -> None:
        current = set(getattr(self, bucket))
        current.add(name)
        setattr(self, bucket, sorted(current))

    def save(self) -> None:
        self.updated_at = datetime.now().isoformat(timespec="seconds")
        (PATHS.checkpoints / "stage2.json").write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls) -> "Stage2Checkpoint":
        p = PATHS.checkpoints / "stage2.json"
        if not p.exists():
            return cls(started_at=datetime.now().isoformat(timespec="seconds"))
        try:
            return cls(**json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:
            stage2_log.warning(f"Unreadable checkpoint ({exc}); starting fresh")
            return cls(started_at=datetime.now().isoformat(timespec="seconds"))


def run_stage2_parsing(cfg: Stage2Config) -> Dict[str, Any]:
    """Parse every document in raw_sources into structured JSON."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    stage2_log.info("=" * 70)
    stage2_log.info("STAGE 2: LAYOUT PARSING")
    stage2_log.info("=" * 70)

    documents, _ = discover_documents(cfg)
    if not documents:
        stage2_log.warning("No documents to parse.")
        return {"error": "no documents found"}

    ckpt = Stage2Checkpoint.load()

    todo: List[Path] = []
    for doc in documents:
        already = parsed_json_path(doc).exists()
        if cfg.skip_existing and not cfg.force_reparse and already:
            ckpt.merge("skipped", doc.name)
        else:
            todo.append(doc)

    stage2_log.info(f"To parse: {len(todo)} | already parsed: {len(ckpt.skipped)}")
    if not todo:
        ckpt.save()
        return {
            "total": len(documents),
            "processed": 0,
            "skipped": len(ckpt.skipped),
            "failed": 0,
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {pool.submit(process_document_with_retry, d, cfg): d for d in todo}

        with tqdm(total=len(todo), desc="Parsing documents") as bar:
            for fut in as_completed(futures):
                doc = futures[fut]
                try:
                    result = fut.result()
                    bucket = "processed" if result["status"] == "success" else "failed"
                except Exception as exc:
                    stage2_log.error(f"Worker crashed on {doc.name}: {exc}")
                    bucket = "failed"

                ckpt.merge(bucket, doc.name)
                bar.set_postfix_str(
                    f"{'ok' if bucket == 'processed' else 'FAIL'} {doc.name[:28]}"
                )
                bar.update(1)
                ckpt.save()

                if bucket == "failed" and not cfg.continue_on_error:
                    raise RuntimeError(f"Parsing failed for {doc.name}")

    summary = {
        "total": len(documents),
        "processed": len(ckpt.processed),
        "skipped": len(ckpt.skipped),
        "failed": len(ckpt.failed),
        "failed_files": ckpt.failed,
        "elapsed_minutes": round((time.time() - t0) / 60, 2),
    }

    stage2_log.info("-" * 70)
    for k in ("total", "processed", "skipped", "failed"):
        stage2_log.info(f"  {k:10s}: {summary[k]}")
    stage2_log.info(f"  elapsed   : {summary['elapsed_minutes']} min")
    if ckpt.failed:
        stage2_log.warning(f"  failures  : {', '.join(ckpt.failed[:5])}")

    return summary


def analyse_parsed_outputs() -> Dict[str, Any]:
    """
    Read this before Stage 3. Two numbers matter most:

      1. `images_resolved` vs `images_missing`. If crops are missing, Stage 3
         will produce a corpus with no diagrams in it and will not complain.

      2. The `discarded` count. MinerU tags page headers, footers, page numbers
         and other layout furniture as `discarded`. Stage 3 drops these, but
         it is worth knowing how much of the corpus is furniture.
    """
    files = sorted(PATHS.parsed.glob("*.json"))

    stats: Dict[str, Any] = {
        "documents": 0,
        "items": 0,
        "types": {},
        "by_source_type": {},
        "images_resolved": 0,
        "images_missing": 0,
        "largest": [],
    }

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("status") != "success":
            continue

        stats["documents"] += 1
        items = data["content_list"]
        stats["items"] += len(items)
        stats["largest"].append((data["filename"], len(items)))

        st = data.get("source_type", "unknown")
        stats["by_source_type"][st] = stats["by_source_type"].get(st, 0) + 1

        md = data.get("metadata", {})
        stats["images_resolved"] += md.get("images_resolved", 0)
        stats["images_missing"] += md.get("images_missing", 0)

        for t, n in md.get("content_types", {}).items():
            stats["types"][t] = stats["types"].get(t, 0) + n

    stats["largest"].sort(key=lambda kv: -kv[1])
    return stats
