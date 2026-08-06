# Book Pipeline

An agentic pipeline that turns raw source material — PDFs, slide decks, lecture
recordings — into a full-length, written technical book. One shared model
does everything except speech-to-text: describing figures, tagging concepts,
planning chapters, writing prose, reviewing it, editing it, and cataloguing
what it wrote. Which model that actually is — local weights, Anthropic, Groq,
a self-hosted vLLM/Ollama/whatever server — is a one-line change in `.env`;
no pipeline code knows or cares which provider is behind it. See
[Model provider](#model-provider).

The system runs in three stages, each independently runnable and resumable:

```
raw sources  --[A: ingestion]-->  chunked corpus  --[B: toc]-->  toc.json  --[C: book_writer]-->  manuscript
```

| Stage | Package | Input | Output |
|---|---|---|---|
| **A — Ingestion** | `src/ingestion/` | `data/raw_sources/` (PDFs, slides, audio/video) | a chunked, provenance-tagged corpus in `runs/<run_id>/data/` |
| **B — Table of Contents** | `src/toc/` | Stage A's chunks | `runs/<run_id>/storage/toc.json` (chapters, sections, word budgets) |
| **C — Book Writer** | `src/book_writer/` | `runs/<run_id>/storage/toc.json` | a written manuscript in `runs/<run_id>/output/` |

(`data/raw_sources/` is the one input path that stays fixed across runs — see [Runs](#runs).)

## Quickstart

```bash
pip install -r requirements.txt   # see "Dependencies" below
cp .env.example .env              # defaults to local weights; see "Model provider" to use an API instead

# drop PDFs / slides / audio / video into data/raw_sources/, then:
python -m src.run_pipeline        # run A -> B -> C as one run
```

Or run a stage at a time: `python -m src.run_pipeline_a`, then `_b`, then
`_c`. Each script is safe to re-run: every stage checkpoints its progress and
skips work that is already done. `run_pipeline_c.py` starts with
`RUN_LIMIT = 1` — read the first section it writes and the ledger diff it
produced before raising the limit (or setting it to `None`) and letting the
full run go unattended.

### Runs

Everything a run *generates* — chunks, `toc.json`, the ledger, written
sections, and every stage's log file — is written under
`runs/<run_id>/{data,storage,output,logs}/`, one run id per process (see
`src/run_context.py`). Running `python -m src.run_pipeline` gives A, B, and C
the same run id automatically; running a stage standalone gives it its own.

The run id is auto-generated unless `RUN_ID` is already set in the
environment. To resume a run that stopped partway through (Pipeline C
checkpoints after every section), re-invoke with the same id:

```bash
RUN_ID=20260806_153000_ab12cd python -m src.run_pipeline_c
```

Two things are deliberately *not* run-scoped, because you author them rather
than the pipeline generating them: `data/raw_sources/` (drop files in once,
every run reads the same copy) and `storage/schemas/constitution.json` (hand
-edited once, "edit it before a real run"). See `SOURCES_ROOT` /
`CONSTITUTION_ROOT` in `.env.example`.

In deployment, point `RUNS_ROOT` at an S3-mounted path (goofys, s3fs, an
EFS/FSx mount) and every run's data/storage/output/logs lands directly in
S3, keyed by run id.

## Architecture

### Stage A — Ingestion (`src/ingestion/`)

Five sub-stages, each in its own module, run in order by `run_pipeline_a.py`:

1. **`stage1_speech.py`** — Whisper (`faster-whisper`) transcribes audio/video,
   keeping word-level timestamps for provenance.
2. **`stage2_parsing.py`** — MinerU/Docling parse documents into structured
   layout JSON (text, figures, tables, equations). `.md`/`.txt` sources skip
   this entirely — they're already plain text. Both MinerU and Docling run
   real layout-detection models locally, which can be heavy on a machine
   with no spare GPU; set `PARSER_TYPE=llm` in `.env` to parse with the
   shared model instead (`llm_parsing.py` — renders each page and asks it to
   transcribe, no local model at all). See
   [Model provider](#model-provider).
3. **`stage3_figures.py`** — the shared model (see
   [Model provider](#model-provider)) describes every figure *in the context
   of the surrounding document*, one call per document rather than one call
   per image. Figure crops are kept on disk; descriptions are inlined into
   the text as `[IMAGE fig_xyz]` markers.
4. **`stage4_chunking.py`** — documents are consolidated into one corpus and
   split into ~1,500–2,000 token chunks, each carrying its source document,
   timestamp (if any), and figures.
5. **`stage5_source_index.py`** — chunks are embedded into a Chroma
   collection for semantic retrieval in Stage C.

### Stage B — Table of Contents (`src/toc/`)

Turns the chunk corpus into a curriculum, in eight steps (`toc/run.py`):
load chunks → extract fine-grained concept tags and their relationships →
normalize tags into a canonical vocabulary → discover chapter-level themes →
assign tags to chapters → form sections within each chapter → order chapters
and sections pedagogically → assemble `storage/toc.json` (under the run's
`storage/`, i.e. `runs/<run_id>/storage/toc.json`) with a word budget per
section.

### Stage C — Book Writer (`src/book_writer/`)

A multi-agent write loop (`BookOrchestrator` in `orchestrator.py`) over seven
memory layers, run once per section:

- **L0 Constitution** (`constitution.py`) — fixed style guide and running
  examples, injected into every prompt.
- **L1 Book Ledger** (`ledger.py`) — what the book has defined, claimed,
  promised, and used so far; seeded from Stage B's tag vocabulary.
- **L2 Draft Store** (`draft_store.py`) — finished sections, on disk and in a
  vector index.
- **L3 Source Memory** (`source_memory.py`) — read-only access to Stage A's
  chunks and figures.
- **L4 Context Assembler** (`context_assembler.py`) — turns L0–L3 into one
  token-budgeted prompt per section.
- **L5/L6 Classroom & Conversation Memory** (`memory.py`) — per-section
  working state, discarded once the section ships.

Five agents (`agents/`) share the one model: **Writer** plans and drafts,
**Reviewer** approves the plan before prose is written, **Student** reads the
prose as a first-time reader and raises doubts, **Editor** smooths the raw
steps into one section (code blocks are masked so it can never touch them),
**Archivist** reads the finished section and writes back into the ledger. A
**Continuity Gate** (5 deterministic checks, no LLM call) and an **Edit
Guard** (verifies the Editor didn't drop content) run alongside.

After the last section, `finishing.py` runs five passes: resolve promises,
generate a glossary and index, flag rough transitions between sections, sweep
for contradicting claims, and report source coverage — then assembles
`output/book/manuscript.md` (under the run's `output/`, i.e.
`runs/<run_id>/output/book/manuscript.md`).

## Model provider

Every LLM call in the system — Stage 3's figure description, all of Pipeline
B, all five of Pipeline C's agents — goes through one interface,
`LLMClient` (`src/llm/base.py`). No pipeline module ever imports a provider
SDK directly; they all call `build_llm_client()` (`src/llm/factory.py`),
which reads `LLM_PROVIDER` from `.env` and returns the matching
implementation. Swapping the model behind the entire system is a one-line
`.env` change, never a code change:

| `LLM_PROVIDER` | What it calls | Needs |
|---|---|---|
| `local` (default) | A `transformers` checkpoint, loaded in-process | `torch`, `transformers`, a GPU |
| `anthropic` | The Anthropic API | `ANTHROPIC_API_KEY` |
| `openai` | The OpenAI API | `OPENAI_API_KEY` |
| `groq` | Whatever model Groq hosts | `GROQ_API_KEY` |
| `gemini` | Gemini, via its OpenAI-compatible endpoint | `GEMINI_API_KEY` |
| `vllm` | A self-hosted vLLM server | `LLM_BASE_URL` |
| `openai_compatible` | Anything else speaking that API shape (Ollama, LM Studio, TGI, ...) | `LLM_BASE_URL` |

`anthropic` gets its own client (`providers/anthropic_client.py`) for its
native Messages API shape; `openai`, `groq`, `gemini`, `vllm`, and
`openai_compatible` all share one client (`providers/openai_compatible.py`)
since they all speak the same `/chat/completions` shape — only the
`base_url`, API key, and model id differ. See `.env.example` for every
variable each provider reads.

Speech-to-text (Whisper, Stage 1) is separate and always local — it isn't
part of this abstraction.

**Never commit or paste a real API key anywhere it could be logged** — a
chat transcript, a terminal share, an issue. Treat any key that's been typed
into one of those as compromised and rotate it.

## Configuration

Configuration is layered:

- **Per-run knobs** (target book length, generation temperature, retry
  limits, ...) are Python dataclasses in each package's `setup.py`
  (`Stage1Config`, `PipelineBConfig`, `WriterConfig`, ...) — edit these in
  code, or construct them with different values in the `run_pipeline_*.py`
  scripts.
- **Per-machine / per-secret values** are environment variables, read from
  `.env` (see `.env.example`): which LLM provider and model, its API key (or
  local HF token / device map), the Whisper size, and `RUNS_ROOT` / `RUN_ID`
  (see [Runs](#runs)).

## Directory layout

```
runs/<run_id>/         everything the pipeline generates, one run id per process
  data/                   Stage A's working tree (corpus out)
    transcripts/, parsed/, converted/, figures/, chunks/, source_index/
  storage/                shared between Stages B and C
    toc.json                 Stage B's output
    book_ledger.json         Stage C's Layer 1, updated after every section
    checkpoints/
  output/                 Stage C's output
    sections/                 one markdown file per section
    book/                     manuscript.md, glossary.md, index.md, completeness_report.md
  logs/                   every stage's log file

data/raw_sources/       NOT run-scoped -- drop PDFs / slides / audio / video here
storage/schemas/        NOT run-scoped -- constitution.json, hand-edited
```

`runs/` holds everything a run generates, and can grow large.

## Dependencies

Core: `tqdm`, `tiktoken`, `python-dotenv`, `Pillow`.

Optional, per stage or provider (each skips cleanly and logs a warning if
its dependency is missing, rather than failing the whole run):

- `faster-whisper`, `ffmpeg` — Stage 1 (speech to text)
- `raganything[all]` (MinerU / Docling) — Stage 2, `PARSER_TYPE=mineru`/`docling` (default)
- `pymupdf` — Stage 2, `PARSER_TYPE=llm` only
- `torch`, `transformers`, optionally `flash-attn` — `LLM_PROVIDER=local`
  only (Stage 3 onward)
- `anthropic` — `LLM_PROVIDER=anthropic` only
- `openai` — `LLM_PROVIDER=openai` / `groq` / `gemini` / `vllm` /
  `openai_compatible` (one SDK, since they all speak the same API shape)
- `chromadb`, `sentence-transformers` — Stage 5's source index, and Stage
  C's draft index and repetition detection

## Notebook

The original design and implementation notebook,
`notebook/BookWriterProject.ipynb`, is kept for reference. `src/` is the
maintained, importable version of the same system.
