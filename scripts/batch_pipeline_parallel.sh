#!/usr/bin/env bash
# Parallel batch-run CTU-FlowRAG pipeline over many scheme folders.
#
# Usage:
#   bash scripts/batch_pipeline_parallel.sh  <NUM_SCHEMES>  <JOBS>  [extra-args]
#   NUM_SCHEMES  number of scheme directories to process (use a large number for "all")
#   JOBS         how many parallel python processes to run (recommended = CPU cores)
#   extra-args   any extra flags forwarded to run_pipeline.py (e.g. --tiling-window 4)
#
# Example:
#   bash scripts/batch_pipeline_parallel.sh 99999 8 --tiling-window 4 --fallback-sentences 6

set -euo

NUM_SCHEMES=${1:-99999}
JOBS=${2:-4}
shift 2 || true  # remove processed args
EXTRA_ARGS="$@"

SCHEMES_ROOT="data/raw/schemes"
OUTPUT_ROOT="output/schemes"
mkdir -p "$OUTPUT_ROOT"

echo "Running pipeline for up to $NUM_SCHEMES schemes using $JOBS parallel jobs"

# Build list of scheme dirs
find "$SCHEMES_ROOT" -maxdepth 1 -type d | sort | head -n "$NUM_SCHEMES" > /tmp/scheme_list.txt

# Export PYTHONUNBUFFERED so progress/errors flush immediately
export PYTHONUNBUFFERED=1

# Function to run one scheme (used by xargs)
run_one() {
  dir="$1"
  slug=$(basename "$dir")
  out_dir="${OUTPUT_ROOT}/${slug}"
  mkdir -p "$out_dir"
  echo "[PID $$] $(date +%H:%M:%S) Processing $slug"
  python -m src.pipeline.run_pipeline \
      --scheme-dir "$dir" \
      --output-dir "$out_dir" \
      --skip-images --quiet --no-llm \
      $EXTRA_ARGS  > /dev/null 2>&1 || {
        echo "⚠️  $slug failed" >&2
      }
}

export -f run_one

# ---------------------------------------------------------------------------
# Portable parallel execution using a simple semaphore loop.
# Works on Bash 3.2 (macOS default) which lacks "wait -n".
# ---------------------------------------------------------------------------

MAX_JOBS="$JOBS"

while read -r dir; do
  # If we already have MAX_JOBS running, pause until one finishes
  while (( $(jobs -pr | wc -l) >= MAX_JOBS )); do
    sleep 0.3
  done

  run_one "$dir" &
done < /tmp/scheme_list.txt

# Wait for all background jobs to finish
wait

echo "✅ All schemes processed. Results in $OUTPUT_ROOT" 