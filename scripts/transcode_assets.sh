#!/usr/bin/env bash
# ==============================================================================
# Sparky Extensible Animation Transcoding Pipeline
# ==============================================================================
set -euo pipefail

OUT_DIR="assets/animations"
mkdir -p "$OUT_DIR"

# Downscale to optimal desktop pet size (384x216) at 24 FPS to save memory
TARGET_SCALE="${TARGET_SCALE:-384:216}"

echo "===================================================================="
echo "Sparky: Transcoding Animation Assets to WebP"
echo "Output Directory: $OUT_DIR"
echo "Target Scale: $TARGET_SCALE @ 24fps"
echo "===================================================================="

convert_video() {
    local src="$1"
    local name="$2"
    if [ ! -f "$src" ]; then
        echo "Warning: $src not found, skipping"
        return
    fi
    echo "Processing $name ($src) -> $OUT_DIR/${name}.webp..."
    ffmpeg -y -i "$src" \
        -vf "scale=${TARGET_SCALE},fps=24" \
        -c:v libwebp -lossless 0 -q:v 80 -compression_level 4 -loop 0 \
        "$OUT_DIR/${name}.webp"
    echo "Generated $OUT_DIR/${name}.webp"
}

# 1. Automatically convert any video in videos/*.mp4 (e.g. sit, happy, naughty, angry, sad, smirk...)
for video in videos/*.mp4; do
    if [ -f "$video" ]; then
        name="$(basename "$video" .mp4)"
        convert_video "$video" "$name"
    fi
done

# 2. Check for optional temp videos if present
if [ -d "videos/temp" ]; then
    for video in videos/temp/*.mp4; do
        if [ -f "$video" ]; then
            name="$(basename "$video" .mp4)"
            convert_video "$video" "$name"
        fi
    done
fi

echo "===================================================================="
echo "Done! All animation assets ready in $OUT_DIR"
echo "===================================================================="
