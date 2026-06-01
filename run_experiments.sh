#!/bin/bash
set -e
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCHEMAS="Project2/schemas"
VAL_INPUT="Project2/validation_input.json"
VAL_GOLD="Project2/validation_gold_schema_links.json"

run_eval() {
    local name=$1
    local adapter=$2
    cp -r "$adapter/." adapter/
    python main.py --input "$VAL_INPUT" --output "preds_${name}.json" --schemas_dir "$SCHEMAS" --batch_size 1
    echo "=== $name ==="
    python Project2/eval.py \
        --predictions "preds_${name}.json" \
        --gold "$VAL_GOLD" \
        --schemas_dir "$SCHEMAS" \
        --questions_input "$VAL_INPUT" | grep "Leaderboard\|Table Score\|Column Score"
}

echo "=== Experiment 1: 7 epochs, 2048 tokens ==="
python train_simple.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 7 --batch_size 1 --grad_accum 16 \
    --max_seq_len 2048 --output_dir ./runs/exp1-7ep-2048
run_eval "exp1-7ep-2048" runs/exp1-7ep-2048/adapter

echo "=== Experiment 2: 5 epochs, 2048 tokens, lr=1e-4 ==="
python train_simple.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 5 --batch_size 1 --grad_accum 16 \
    --lr 1e-4 --max_seq_len 2048 --output_dir ./runs/exp2-5ep-lr1e4
run_eval "exp2-5ep-lr1e4" runs/exp2-5ep-lr1e4/adapter

echo "=== Experiment 3: 8 epochs, 2048 tokens, r=16 ==="
python train_simple.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 8 --batch_size 1 --grad_accum 16 \
    --lora_r 16 --max_seq_len 2048 --output_dir ./runs/exp3-8ep-r16
run_eval "exp3-8ep-r16" runs/exp3-8ep-r16/adapter

echo "=== Experiment 4: 5 epochs, 2048 tokens, 0.5B model ==="
python train_simple.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --epochs 5 --batch_size 2 --grad_accum 8 \
    --max_seq_len 2048 --output_dir ./runs/exp4-0.5b
run_eval "exp4-0.5b" runs/exp4-0.5b/adapter

echo "All experiments done."
echo "Results summary:"
for name in exp1-7ep-2048 exp2-5ep-lr1e4 exp3-8ep-r16 exp4-0.5b; do
    echo -n "$name: "
    python Project2/eval.py \
        --predictions "preds_${name}.json" \
        --gold "$VAL_GOLD" \
        --schemas_dir "$SCHEMAS" \
        --questions_input "$VAL_INPUT" 2>/dev/null | grep "Leaderboard"
done
