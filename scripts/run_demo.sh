#!/bin/bash
set -e

ONNX="models/sign_model.onnx"

echo ""
echo "=== ASL Real-Time Demo ==="
echo ""

# ── Model selection ──────────────────────────────────────────────────────────
DEMO_DIR="models/demo"

# Collect .pt files from models/demo/
MODEL_FILES=()
while IFS= read -r line; do
    MODEL_FILES+=("$line")
done < <(find "$DEMO_DIR" -maxdepth 1 -name "*.pt" | sort)

if [ ${#MODEL_FILES[@]} -eq 0 ]; then
    echo "No models found in $DEMO_DIR."
    echo "Place .pt checkpoint files there, or using the default checkpoint."
    echo ""
    CHECKPOINT="models/checkpoints/transformer_d128_l3_v1896_noaug_combined_best.pt"
    VOCAB=1896
else
    echo "Select a model:"
    for i in "${!MODEL_FILES[@]}"; do
        fname=$(basename "${MODEL_FILES[$i]}")
        # Try to parse vocab size from filename (e.g. _v1896_)
        vocab_hint=$(echo "$fname" | grep -oE 'v[0-9]+' | head -1 | tr -d 'v')
        [ -n "$vocab_hint" ] && label="  vocab ~$vocab_hint" || label=""
        printf "  %d) %s%s\n" $((i+1)) "$fname" "$label"
    done
    echo ""
    read -p "Choice [1-${#MODEL_FILES[@]}]: " model_choice

    if ! [[ "$model_choice" =~ ^[0-9]+$ ]] || \
       [ "$model_choice" -lt 1 ] || [ "$model_choice" -gt ${#MODEL_FILES[@]} ]; then
        echo "Invalid choice, using model 1."
        model_choice=1
    fi

    CHECKPOINT="${MODEL_FILES[$((model_choice-1))]}"

    # Parse vocab from filename; ask if not found
    VOCAB=$(basename "$CHECKPOINT" | grep -oE 'v[0-9]+' | head -1 | tr -d 'v')
    if [ -z "$VOCAB" ]; then
        read -p "Vocab size not detected in filename. Enter vocab size: " VOCAB
    fi
fi

echo ""
echo "Model:  $CHECKPOINT"
echo "Vocab:  $VOCAB"
echo ""

# ── Threshold selection ──────────────────────────────────────────────────────
echo "Select confidence threshold:"
echo "  1) 0.05  — show almost everything"
echo "  2) 0.10  — low filter (default)"
echo "  3) 0.20  — medium filter"
echo "  4) 0.40  — high filter (stricter)"
echo "  5) Custom"
echo ""
read -p "Choice [1-5]: " choice

case $choice in
    1) THRESHOLD=0.05 ;;
    2) THRESHOLD=0.10 ;;
    3) THRESHOLD=0.20 ;;
    4) THRESHOLD=0.40 ;;
    5)
        read -p "Enter threshold (0.0 - 1.0): " THRESHOLD
        ;;
    *)
        echo "Invalid choice, using default 0.10"
        THRESHOLD=0.10
        ;;
esac

echo ""
echo "Threshold: $THRESHOLD"
echo ""

# ── Export & run ─────────────────────────────────────────────────────────────
echo "--- Exporting to ONNX ---"
python -m src.export --checkpoint "$CHECKPOINT" --vocab $VOCAB --output "$ONNX"

echo ""
echo "--- Starting demo (press Q to quit) ---"
echo ""
python -m src.demo --model "$ONNX" --vocab $VOCAB --threshold $THRESHOLD
