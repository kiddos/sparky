#!/usr/bin/env bash
# ==============================================================================
# Download Google Gemma 3 270M GGUF (Quantized) for Sparky Local Inference
# ==============================================================================
set -euo pipefail

DEST_TARGET="${1:-models/gemma-3-270m/gemma-3-270m-it-q4_k_m.gguf}"

# If destination target is an existing directory or ends with a slash, append default filename
if [ -d "$DEST_TARGET" ] || [[ "$DEST_TARGET" == */ ]]; then
    DEST_FILE="${DEST_TARGET%/}/gemma-3-270m-it-q4_k_m.gguf"
else
    DEST_FILE="$DEST_TARGET"
fi

DEST_DIR="$(dirname "$DEST_FILE")"
mkdir -p "$DEST_DIR"

echo "===================================================================="
echo "Sparky: Downloading Gemma 3 270M Q4_K_M GGUF (~241 MB)"
echo "Destination: $DEST_FILE"
echo "===================================================================="

# Validation helper: check that the file exists, has plausible size (>50MB),
# and starts with the standard GGUF magic header ('GGUF')
verify_gguf() {
    local target="$1"
    if [ ! -f "$target" ]; then
        return 1
    fi
    local size
    size=$(wc -c < "$target" 2>/dev/null || echo 0)
    # The Q4_K_M model is ~253 MB (253,115,424 bytes). Must be at least 50 MB.
    if [ "$size" -lt 50000000 ]; then
        echo "Verification failed: file size (${size} bytes) is suspiciously small." >&2
        return 1
    fi
    local magic
    magic=$(head -c 4 "$target" 2>/dev/null || true)
    if [ "$magic" != "GGUF" ]; then
        echo "Verification failed: file header does not match GGUF magic bytes (got '${magic}')." >&2
        return 1
    fi
    return 0
}

# If the file already exists and passes verification, no re-download is required unless --force is given
if [ -f "$DEST_FILE" ] && [ "${2:-}" != "--force" ]; then
    if verify_gguf "$DEST_FILE"; then
        echo "Model already exists and is verified at: $DEST_FILE"
        echo "Run Sparky locally: cargo run --release"
        exit 0
    else
        echo "Existing file at $DEST_FILE is invalid or corrupt. Re-downloading..."
        rm -f "$DEST_FILE"
    fi
fi

# Define primary and fallback mirrors
# Primary: unsloth/gemma-3-270m-it-GGUF (public, high-speed, direct Hugging Face host)
# Fallback: bartowski/google_gemma-3-270m-it-GGUF (public mirror)
MIRRORS=(
    "unsloth/gemma-3-270m-it-GGUF|gemma-3-270m-it-Q4_K_M.gguf|https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf"
    "bartowski/google_gemma-3-270m-it-GGUF|google_gemma-3-270m-it-Q4_K_M.gguf|https://huggingface.co/bartowski/google_gemma-3-270m-it-GGUF/resolve/main/google_gemma-3-270m-it-Q4_K_M.gguf"
)

# Auth token support if configured by user
AUTH_HEADER=()
WGET_HEADER=()
TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [ -n "$TOKEN" ]; then
    AUTH_HEADER=(-H "Authorization: Bearer $TOKEN")
    WGET_HEADER=(--header="Authorization: Bearer $TOKEN")
fi

TMP_FILE="${DEST_FILE}.tmp.$$"
cleanup() {
    rm -f "$TMP_FILE"
}
trap cleanup EXIT INT TERM

DOWNLOADED=0

for MIRROR in "${MIRRORS[@]}"; do
    IFS="|" read -r REPO FILENAME URL <<< "$MIRROR"
    echo "Trying source: $REPO ($FILENAME)..."

    # 1. Try curl if available
    if command -v curl >/dev/null 2>&1; then
        echo "Downloading with curl: $URL..."
        if curl -f -L --show-error --progress-bar "${AUTH_HEADER[@]}" "$URL" -o "$TMP_FILE"; then
            if verify_gguf "$TMP_FILE"; then
                mv -f "$TMP_FILE" "$DEST_FILE"
                DOWNLOADED=1
                break
            else
                rm -f "$TMP_FILE"
            fi
        else
            echo "curl download failed for $URL" >&2
            rm -f "$TMP_FILE"
        fi
    fi

    # 2. Try wget if available
    if command -v wget >/dev/null 2>&1; then
        echo "Downloading with wget: $URL..."
        if wget "${WGET_HEADER[@]}" --progress=bar:force -O "$TMP_FILE" "$URL"; then
            if verify_gguf "$TMP_FILE"; then
                mv -f "$TMP_FILE" "$DEST_FILE"
                DOWNLOADED=1
                break
            else
                rm -f "$TMP_FILE"
            fi
        else
            echo "wget download failed for $URL" >&2
            rm -f "$TMP_FILE"
        fi
    fi

    # 3. Try huggingface-cli if available
    if command -v huggingface-cli >/dev/null 2>&1; then
        echo "Downloading with huggingface-cli: $REPO / $FILENAME..."
        if huggingface-cli download "$REPO" "$FILENAME" --local-dir "$DEST_DIR"; then
            DOWNLOADED_PATH="$DEST_DIR/$FILENAME"
            if [ -f "$DOWNLOADED_PATH" ] && verify_gguf "$DOWNLOADED_PATH"; then
                if [ "$DOWNLOADED_PATH" != "$DEST_FILE" ]; then
                    mv -f "$DOWNLOADED_PATH" "$DEST_FILE"
                fi
                DOWNLOADED=1
                break
            fi
        else
            echo "huggingface-cli download failed for $REPO" >&2
        fi
    fi

    # 4. Try python3 urllib if available
    if command -v python3 >/dev/null 2>&1; then
        echo "Downloading with python3 urllib: $URL..."
        if python3 -c '
import sys, urllib.request, os

url = sys.argv[1]
dest = sys.argv[2]
token = sys.argv[3]

headers = {"User-Agent": "sparky-pet/0.1"}
if token:
    headers["Authorization"] = f"Bearer {token}"

req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
    total = int(resp.headers.get("Content-Length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024 # 1 MB
    while True:
        chunk = resp.read(chunk_size)
        if not chunk:
            break
        f.write(chunk)
        downloaded += len(chunk)
        if total > 0:
            percent = (downloaded / total) * 100
            print(f"\rProgress: {downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB ({percent:.1f}%)", end="", flush=True)
    print("")
' "$URL" "$TMP_FILE" "$TOKEN"; then
            if verify_gguf "$TMP_FILE"; then
                mv -f "$TMP_FILE" "$DEST_FILE"
                DOWNLOADED=1
                break
            else
                rm -f "$TMP_FILE"
            fi
        else
            echo "python3 download failed for $URL" >&2
            rm -f "$TMP_FILE"
        fi
    fi
done

if [ "$DOWNLOADED" -eq 1 ]; then
    echo ""
    echo "===================================================================="
    echo "Success! Gemma 3 270M Q4_K_M GGUF model ready at: $DEST_FILE"
    echo "Run Sparky locally: cargo run --release"
    echo "===================================================================="
    exit 0
fi

echo ""
echo "Error: Failed to download Gemma 3 270M GGUF model from available mirrors." >&2
echo ""
echo "To download manually:"
echo "1. Download https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf"
echo "2. Save to $DEST_FILE"
echo "3. Run Sparky: cargo run --release"
exit 1
