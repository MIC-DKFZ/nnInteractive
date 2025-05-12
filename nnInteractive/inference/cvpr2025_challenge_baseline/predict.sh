#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# CVPR25 – Foundation Models for Interactive 3D Biomedical Image Segmentation
# Skeleton inference script executed by the evaluation server.
# -----------------------------------------------------------------------------
# EXPECTED ENVIRONMENT  (mounted by the evaluation script)
#   /workspace/inputs   – *.npz files containing “imgs” (+ prompts, etc.)
#   /workspace/outputs  – write one *.npz with key "segs" for each input case
# -----------------------------------------------------------------------------
set -e
echo "[predict.sh] Starting inference …"

INPUT_DIR=/workspace/inputs
OUTPUT_DIR=/workspace/outputs

# Safety checks --------------------------------------------------------------
if [ ! -d "$INPUT_DIR" ];  then
  echo "[predict.sh] ERROR: $INPUT_DIR does not exist."
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

# Loop over every .npz in inputs --------------------------------------------
for CASE_PATH in "$INPUT_DIR"/*.npz ; do
  CASE_FILE=$(basename "$CASE_PATH")
  echo "[predict.sh] -> Processing $CASE_FILE"
  python /workspace/predict.py \
      --case_path "$CASE_PATH" \
      --save_path "$OUTPUT_DIR/$CASE_FILE"
done

echo "[predict.sh] Inference completed."