#!/usr/bin/env bash
# ==============================================================================
# Download Google Gemma 3 270M for Sparky Local Inference
# ==============================================================================
set -euo pipefail

DEST_DIR="${1:-models/gemma-3-270m}"
mkdir -p "$DEST_DIR"

echo "===================================================================="
echo "Sparky: Downloading Gemma 3 270M (google/gemma-3-270m-it)"
echo "Destination: $DEST_DIR"
echo "===================================================================="

# Check for huggingface-cli
if command -v huggingface-cli >/dev/null 2>&1; then
    echo "Using huggingface-cli..."
    huggingface-cli download google/gemma-3-270m-it \
        --local-dir "$DEST_DIR" \
        --include "config.json" "tokenizer.json" "model.safetensors"
    echo ""
    echo "Done! Gemma 3 270M model files are ready in: $DEST_DIR"
    echo "Run Sparky locally: cargo run --release -- --local"
    exit 0
fi

# Fallback: using python -c "from huggingface_hub import snapshot_download..."
if command -v python3 >/dev/null 2>&1; then
    echo "huggingface-cli not in PATH, attempting python huggingface_hub..."
    if python3 -c "import huggingface_hub" >/dev/null 2>&1; then
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-3-270m-it', local_dir='$DEST_DIR', allow_patterns=['config.json', 'tokenizer.json', 'model.safetensors'])"
        echo ""
        echo "Done! Gemma 3 270M model files are ready in: $DEST_DIR"
        echo "Run Sparky locally: cargo run --release -- --local"
        exit 0
    fi
fi

# Manual instructions if automated tools are missing
echo "Automated downloader tool (huggingface-cli or python huggingface_hub) not detected."
echo ""
echo "To download manually:"
echo "1. Install huggingface_hub:"
echo "   pip install -U huggingface_hub"
echo "2. Download the Gemma 3 270M files:"
echo "   huggingface-cli download google/gemma-3-270m-it --local-dir $DEST_DIR --include \"config.json\" \"tokenizer.json\" \"model.safetensors\""
echo "3. Run Sparky with local inference:"
echo "   cargo run --release -- --local"
exit 1
