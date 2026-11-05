#!/usr/bin/env bash
# ==============================================================================
# Download Google Gemma 3 270M GGUF (Quantized) for Sparky Local Inference
# ==============================================================================
set -euo pipefail

DEST_FILE="${1:-models/gemma-3-270m/gemma-3-270m-it-q4_k_m.gguf}"
DEST_DIR="$(dirname "$DEST_FILE")"
mkdir -p "$DEST_DIR"

echo "===================================================================="
echo "Sparky: Downloading Gemma 3 270M Q4_K_M GGUF (~165 MB)"
echo "Destination: $DEST_FILE"
echo "===================================================================="

# HF Repository containing Gemma 3 GGUF quants
HF_REPO="google/gemma-3-270m-it-GGUF"
FILENAME="gemma-3-270m-it-Q4_K_M.gguf"
HF_DIRECT_URL="https://huggingface.co/${HF_REPO}/resolve/main/${FILENAME}"

# 1. Try huggingface-cli
if command -v huggingface-cli >/dev/null 2>&1; then
    echo "Using huggingface-cli..."
    huggingface-cli download "$HF_REPO" "$FILENAME" --local-dir "$DEST_DIR"
    if [ -f "$DEST_DIR/$FILENAME" ] && [ "$DEST_DIR/$FILENAME" != "$DEST_FILE" ]; then
        mv "$DEST_DIR/$FILENAME" "$DEST_FILE"
    fi
    echo ""
    echo "Done! Gemma 3 270M Q4_K_M GGUF model ready at: $DEST_FILE"
    echo "Run Sparky locally: cargo run --release"
    exit 0
fi

# 2. Try curl
if command -v curl >/dev/null 2>&1; then
    echo "Using curl to download from Hugging Face..."
    curl -L "$HF_DIRECT_URL" -o "$DEST_FILE"
    echo ""
    echo "Done! Gemma 3 270M Q4_K_M GGUF model ready at: $DEST_FILE"
    echo "Run Sparky locally: cargo run --release"
    exit 0
fi

# 3. Try wget
if command -v wget >/dev/null 2>&1; then
    echo "Using wget to download from Hugging Face..."
    wget -O "$DEST_FILE" "$HF_DIRECT_URL"
    echo ""
    echo "Done! Gemma 3 270M Q4_K_M GGUF model ready at: $DEST_FILE"
    echo "Run Sparky locally: cargo run --release"
    exit 0
fi

# 4. Fallback: using python huggingface_hub
if command -v python3 >/dev/null 2>&1; then
    echo "curl/wget not found, attempting python huggingface_hub..."
    if python3 -c "import huggingface_hub" >/dev/null 2>&1; then
        python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='$HF_REPO', filename='$FILENAME', local_dir='$DEST_DIR')"
        if [ -f "$DEST_DIR/$FILENAME" ] && [ "$DEST_DIR/$FILENAME" != "$DEST_FILE" ]; then
            mv "$DEST_DIR/$FILENAME" "$DEST_FILE"
        fi
        echo ""
        echo "Done! Gemma 3 270M Q4_K_M GGUF model ready at: $DEST_FILE"
        echo "Run Sparky locally: cargo run --release"
        exit 0
    fi
fi

echo "Error: Neither huggingface-cli, curl, wget, nor python huggingface_hub was detected."
echo ""
echo "To download manually:"
echo "1. Download $HF_DIRECT_URL"
echo "2. Save to $DEST_FILE"
echo "3. Run Sparky: cargo run --release"
exit 1
