#!/bin/bash
# 8-hour final push — maximize score before submission
# tmux new -s eight && conda activate cse234 && cd ~/CSE234_proj2
# bash run_8hour.sh 2>&1 | tee eight_hour.log

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
    cp -r "runs/$name/adapter/." adapter/
    python main.py --input $VAL_INPUT --output "preds_${name}.json" \
        --schemas_dir $SCHEMAS --batch_size 4 --two_stage
    run_eval "$name" "preds_${name}.json"
    echo "$(date): $name done"
}

echo "$(date): Starting 8-hour final push experiments"
echo "Current best: 0.4432 (final-B-r32)"

# === TIER 1: Direct improvements on current best ===

# E1: r=32 + more epochs (7) — might close the gap
train_and_eval "e1-r32-7ep" \
    --train augmented_data/train_v2.json \
    --epochs 7 --batch_size 1 --grad_accum 16 --lr 1e-4 --max_seq_len 2048 \
    --show_fk_links --sort_by_question --lora_r 32

# E2: r=32 + oversample 2x (more weight on hard SAP/NTSB)
train_and_eval "e2-r32-oversample" \
    --train augmented_data/train_v2.json \
    --epochs 5 --batch_size 1 --grad_accum 16 --lr 1e-4 --max_seq_len 2048 \
    --show_fk_links --sort_by_question --lora_r 32 --oversample_factor 2

# E3: r=64 — maximum LoRA capacity
train_and_eval "e3-r64" \
    --train augmented_data/train_v2.json \
    --epochs 5 --batch_size 1 --grad_accum 16 --lr 1e-4 --max_seq_len 2048 \
    --show_fk_links --sort_by_question --lora_r 64

# E4: r=32 + slightly lower LR (less aggressive, better generalization)
train_and_eval "e4-r32-lr8e5" \
    --train augmented_data/train_v2.json \
    --epochs 6 --batch_size 1 --grad_accum 16 --lr 8e-5 --max_seq_len 2048 \
    --show_fk_links --sort_by_question --lora_r 32

# === TIER 2: Architecture changes ===

# E5: Qwen2.5-0.5B but many more epochs (fast model, try to get more iterations)
train_and_eval "e5-05b-10ep" \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --train augmented_data/train_v2.json \
    --epochs 10 --batch_size 2 --grad_accum 8 --lr 2e-4 --max_seq_len 2048 \
    --show_fk_links --sort_by_question --lora_r 32

# E6: r=32 + target all linear layers (more expressive)
train_and_eval "e6-r32-alllinear" \
    --train augmented_data/train_v2.json \
    --epochs 5 --batch_size 1 --grad_accum 16 --lr 1e-4 --max_seq_len 2048 \
    --show_fk_links --sort_by_question --lora_r 32 --lora_all_linear

# E7: r=32 + 6 epochs + best settings
train_and_eval "e7-r32-6ep-best" \
    --train augmented_data/train_v2.json \
    --epochs 6 --batch_size 1 --grad_accum 16 --lr 1e-4 --max_seq_len 2048 \
    --show_fk_links --sort_by_question --lora_r 32 --oversample_factor 2

# === FINAL SUMMARY ===
echo ""; echo "======== FINAL SUMMARY ========"
echo "current_best: 0.4432 (final-B-r32)"
for name in e1-r32-7ep e2-r32-oversample e3-r64 e4-r32-lr8e5 e5-05b-10ep e6-r32-alllinear e7-r32-6ep-best; do
    [ -f "preds_${name}.json" ] && run_eval "$name" "preds_${name}.json" || echo "$name: not completed"
done
echo "$(date): All done"
