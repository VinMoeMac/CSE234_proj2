#!/bin/bash
# Final push: E4 variants + stage3 multi-table
# tmux new -s push && conda activate cse234 && cd ~/CSE234_proj2
# bash run_final_push.sh 2>&1 | tee final_push.log

set -e
SCHEMAS="Project2/schemas"
VAL_INPUT="Project2/validation_input.json"
VAL_GOLD="Project2/validation_gold_schema_links.json"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

run_eval() {
    local name=$1; local pred=$2
    echo -n "$name: "
    python Project2/eval.py --predictions "$pred" --gold $VAL_GOLD \
        --schemas_dir $SCHEMAS --questions_input $VAL_INPUT 2>/dev/null \
        | grep "Leaderboard\|Table Score\|Column Score"
}

train_and_eval() {
    local name=$1; shift
    echo ""; echo "======== $(date): $name ========"
    python train_simple.py --model $MODEL --validation Project2/validation.json \
        --schemas_dir $SCHEMAS --output_dir "./runs/$name" "$@"
    python main.py --input $VAL_INPUT --output "preds_${name}.json" \
        --schemas_dir $SCHEMAS --adapter_path "./runs/$name/adapter" \
        --batch_size 4 --two_stage
    run_eval "$name" "preds_${name}.json"
    echo "$(date): $name done"
}

echo "$(date): Starting final push — current best: 0.4539 (e4-r32-lr8e5)"

# === Part 1: E4 variants ===

# P1: lr=6e-5 (between 8e-5 and 5e-5)
train_and_eval "p1-r32-lr6e5" \
    --train augmented_data/train_v2.json \
    --epochs 6 --batch_size 1 --grad_accum 16 --lr 6e-5 --max_seq_len 2048 \
    --show_fk_links --sort_by_question --lora_r 32

# P2: lr=8e-5 + 8 epochs (more training with same LR)
train_and_eval "p2-r32-lr8e5-8ep" \
    --train augmented_data/train_v2.json \
    --epochs 8 --batch_size 1 --grad_accum 16 --lr 8e-5 --max_seq_len 2048 \
    --show_fk_links --sort_by_question --lora_r 32

# P3: lr=5e-5 + 8 epochs (even lower LR, more epochs)
train_and_eval "p3-r32-lr5e5-8ep" \
    --train augmented_data/train_v2.json \
    --epochs 8 --batch_size 1 --grad_accum 16 --lr 5e-5 --max_seq_len 2048 \
    --show_fk_links --sort_by_question --lora_r 32

# === Part 2: Stage 3 — multi-table query detection ===
# Use the best adapter from Part 1 (whichever scores highest)
# Stage 3 asks: "does this question need multiple tables?" and if so
# runs stage2 on top-3 tables instead of just predicted ones

echo ""; echo "======== $(date): Testing stage3 with best E4 adapter ========"
# Use e4 adapter (current best)
cp -r runs/e4-r32-lr8e5/adapter/. adapter/
python main.py --input $VAL_INPUT --output preds_e4_topk3.json \
    --schemas_dir $SCHEMAS --adapter_path runs/e4-r32-lr8e5/adapter \
    --batch_size 4 --two_stage --two_stage_topk 2
run_eval "e4+topk2" preds_e4_topk3.json

# Also test best p-experiment if it beat e4
echo ""; echo "======== SUMMARY ========"
echo "e4-r32-lr8e5 (current best): 0.4539"
for name in p1-r32-lr6e5 p2-r32-lr8e5-8ep p3-r32-lr5e5-8ep; do
    [ -f "preds_${name}.json" ] && run_eval "$name" "preds_${name}.json" || echo "$name: not done"
done
run_eval "e4+topk2" preds_e4_topk3.json

echo "$(date): Done"
