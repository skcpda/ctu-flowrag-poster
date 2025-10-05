#!/bin/bash
# Training script for RCR-GAT model

set -e

# Default values
CONFIG="ctu_flowrag/configs/rcr_gat.yaml"
DOC_INDEX="data/dev_index.json"
TENSOR_DIR="data/tensors"
OUTPUT_DIR="logs"
DEVICE="auto"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --doc_index)
            DOC_INDEX="$2"
            shift 2
            ;;
        --tensor_dir)
            TENSOR_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --config CONFIG        Path to config file (default: ctu_flowrag/configs/rcr_gat.yaml)"
            echo "  --doc_index INDEX      Path to document index file (default: data/dev_index.json)"
            echo "  --tensor_dir DIR       Path to tensor directory (default: data/tensors)"
            echo "  --output_dir DIR       Output directory (default: logs)"
            echo "  --device DEVICE        Device to use (default: auto)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if required files exist
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

if [[ ! -f "$DOC_INDEX" ]]; then
    echo "Error: Document index file not found: $DOC_INDEX"
    exit 1
fi

if [[ ! -d "$TENSOR_DIR" ]]; then
    echo "Error: Tensor directory not found: $TENSOR_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
echo "Starting RCR-GAT training..."
echo "Config: $CONFIG"
echo "Document index: $DOC_INDEX"
echo "Tensor directory: $TENSOR_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"

python -m ctu_flowrag.train.train_rcr_gat \
    --config "$CONFIG" \
    --doc_index "$DOC_INDEX" \
    --tensor_dir "$TENSOR_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

echo "Training completed!"

