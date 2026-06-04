#!/bin/bash
# Final overnight experiments targeting 0.50+
# Run in tmux: tmux new -s final && conda activate cse234 && cd ~/CSE234_proj2
# Then: bash run_final_overnight.sh 2>&1 | tee final_overnight.log

set -e
SCHEMAS="Project2/schemas"
VAL_INPUT="Project2/validation_input.json"
VAL_GOLD="Project2/validation_gold_schema_links.json"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

run_eval() {
    local name=$1; local pred=$2
    echo -n "$name: "
    python Project2/eval.py --predictions "$pred" --gold $VAL_GOLD \
        --schemas_dir $SCHEMAS --questions_input $VAL_INPUT 2>/dev/null | grep Leaderboard
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
}

echo "$(date): Starting final overnight experiments"

# Exp A: v2 synthetic (NTSB+SAP targeted) + FK + sorting + two-stage
train_and_eval "final-A-v2synth" \
    --train augmented_data/train_v2.json \
    --epochs 5 --batch_size 1 --grad_accum 16 --lr 1e-4 --max_seq_len 2048 \
    --show_fk_links --sort_by_question

# Exp B: v2 synthetic + bigger LoRA (r=32)
train_and_eval "final-B-r32" \
    --train augmented_data/train_v2.json \
    --epochs 5 --batch_size 1 --grad_accum 16 --lr 1e-4 --max_seq_len 2048 \
    --show_fk_links --sort_by_question --lora_r 32

# Exp C: v2 synthetic + lower LR for better generalization
train_and_eval "final-C-lr5e5" \
    --train augmented_data/train_v2.json \
    --epochs 7 --batch_size 1 --grad_accum 16 --lr 5e-5 --max_seq_len 2048 \
    --show_fk_links --sort_by_question

# Exp D: all synthetic combined (v1 59 + v2 46 = 105 extra) + FK + sorting
train_and_eval "final-D-allsynth" \
    --train augmented_data/train_v2.json \
    --epochs 5 --batch_size 1 --grad_accum 16 --lr 1e-4 --max_seq_len 2048 \
    --show_fk_links --sort_by_question --oversample_factor 2

echo ""; echo "======== SUMMARY ========"
echo -n "current_best (camel-split-v1 + 2stage 0.440): "
echo "0.4400"
for name in final-A-v2synth final-B-r32 final-C-lr5e5 final-D-allsynth; do
    [ -f "preds_${name}.json" ] && run_eval "$name" "preds_${name}.json"
done
echo "$(date): Done"
