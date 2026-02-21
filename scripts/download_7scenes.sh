#!/usr/bin/env bash
set -euo pipefail

# Download and extract the 7Scenes dataset (Microsoft Research).
#
# Usage:
#   ./scripts/download_7scenes.sh [output_dir] [scene1 scene2 ...]
#   ./scripts/download_7scenes.sh data/7scenes chess fire
#   ./scripts/download_7scenes.sh data/7scenes                # downloads all

OUTPUT_DIR="${1:-data/7scenes}"
shift 2>/dev/null || true

ALL_SCENES="chess fire heads office pumpkin redkitchen stairs"
SCENES="${@:-$ALL_SCENES}"

BASE_URL="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"

mkdir -p "$OUTPUT_DIR"

for scene in $SCENES; do
    scene_dir="$OUTPUT_DIR/$scene"

    if [ -d "$scene_dir" ] && [ "$(ls -A "$scene_dir" 2>/dev/null)" ]; then
        echo "Scene '$scene' already exists at $scene_dir, skipping."
        continue
    fi

    rm -rf "$scene_dir"
    zip_file="$OUTPUT_DIR/${scene}.zip"

    echo "Downloading $scene..."
    curl -L -o "$zip_file" "$BASE_URL/$scene.zip"

    echo "Extracting $scene..."
    unzip -q -o "$zip_file" -d "$OUTPUT_DIR"
    rm "$zip_file"

    echo "Done: $scene -> $scene_dir"
    du -sh "$scene_dir"
    echo
done

echo "7Scenes download complete."
echo "Location: $OUTPUT_DIR"
echo "Scenes: $SCENES"
