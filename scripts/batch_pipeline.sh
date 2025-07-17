#!/usr/bin/env bash
# Batch-run CTU-FlowRAG pipeline over many scheme folders.
#
# Usage:
#     bash scripts/batch_pipeline.sh 150   # process first 150 schemes
#
# It creates one output directory per scheme under output/schemes/<slug>/
# so that results do not overwrite each other, then runs the full pipeline
# (without LLM role-tagger by default for speed).

set -euo pipefail

NUM_SCHEMES=${1:-150}
SCHEMES_ROOT="data/raw/schemes"
OUTPUT_ROOT="output/schemes"

mkdir -p "$OUTPUT_ROOT"

# Use existing OPENAI_API_KEY from environment if set; do not override.

# Enumerate scheme directories (sorted for reproducibility)
shift || true  # shift off NUM_SCHEMES if provided
EXTRA_ARGS="$@"

count=0
for dir in $(ls -1d "$SCHEMES_ROOT"/* | head -n "$NUM_SCHEMES"); do
    slug=$(basename "$dir")
    out_dir="${OUTPUT_ROOT}/${slug}"
    mkdir -p "$out_dir"

    echo "[ $((count+1)) / $NUM_SCHEMES ] Running pipeline for $slug …"

    python -m src.pipeline.run_pipeline \
        --scheme-dir "$dir" \
        --output-dir "$out_dir" \
        $EXTRA_ARGS \
        --no-llm || {
        echo "⚠️  Pipeline failed for $slug – skipping" >&2
    }

    ((count++))
    echo "------------------------------------------------------------"
done

echo "✅ Completed $count / $NUM_SCHEMES schemes. Results in $OUTPUT_ROOT" 