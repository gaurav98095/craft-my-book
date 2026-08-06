"""
Master runner — Pipeline A, then B, then C, sharing one run.

    python -m src.run_pipeline

Every stage's logs, checkpoints, and generated artifacts land under
runs/<run_id>/{data,storage,output,logs}/ — see run_context.py. The run id
is auto-generated unless RUN_ID is already set in the environment.

To resume a run that crashed partway through (Pipeline C checkpoints after
every section), re-invoke with the same RUN_ID so it picks the same folder
back up:

    RUN_ID=20260806_153000_ab12cd python -m src.run_pipeline
"""

# Import order matters: this sets RUN_ID and the DATA_ROOT / STORAGE_ROOT /
# OUTPUT_ROOT / LOGS_ROOT env vars before run_pipeline_a/b/c (and the setup
# modules they import) construct their path objects, which is what makes all
# three pipelines share one run when run together like this.
from . import run_context

from .run_pipeline_a import main as run_pipeline_a
from .run_pipeline_b import main as run_pipeline_b
from .run_pipeline_c import main as run_pipeline_c


def main() -> None:
    print("=" * 72)
    print(f"RUN {run_context.RUN_ID}")
    print(f"  data    -> {run_context.RUN_ROOT / 'data'}")
    print(f"  storage -> {run_context.RUN_ROOT / 'storage'}")
    print(f"  output  -> {run_context.RUN_ROOT / 'output'}")
    print(f"  logs    -> {run_context.RUN_ROOT / 'logs'}")
    print("=" * 72)

    print("\n>>> PIPELINE A — ingestion\n")
    run_pipeline_a()

    print("\n>>> PIPELINE B — table of contents\n")
    run_pipeline_b()

    print("\n>>> PIPELINE C — book writer\n")
    run_pipeline_c()

    print("=" * 72)
    print(f"RUN {run_context.RUN_ID} COMPLETE — {run_context.RUN_ROOT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
