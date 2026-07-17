#!/usr/bin/env bash
set -euo pipefail

# Cho phép gọi script từ bất kỳ cwd nào (mọi lệnh bên dưới đều dùng đường
# dẫn tương đối final/...) — chuyển về thư mục gốc repo trước tiên.
cd "$(dirname "$0")/.."

# PY có thể override: PY=~/work/.venv/bin/python ./final/run_all.sh
PY="${PY:-.venv/bin/python}"

# Tham số đầu tiên là --smoke thì chạy nhanh (512 mẫu, 2 epoch) cho cả 3 run
# train — dùng để kiểm tra toàn bộ pipeline trước khi chạy thật trên trainbox.
SMOKE_FLAG=""
SUFFIX=""
EVAL_LIMIT=()          # smoke: chỉ chấm 32 ảnh test cho nhanh (beam search chậm)
if [ "${1:-}" = "--smoke" ]; then
    SMOKE_FLAG="--smoke"
    SUFFIX="_smoke"
    EVAL_LIMIT=(--limit 32)
fi

RUN_LSTM="lstm${SUFFIX}"
RUN_TRANSFORMER="transformer${SUFFIX}"
RUN_LSTM_NOATTN="lstm_noattn${SUFFIX}"

CKPT_DIR="final/checkpoints"
CKPT_LSTM="${CKPT_DIR}/${RUN_LSTM}.pt"
CKPT_TRANSFORMER="${CKPT_DIR}/${RUN_TRANSFORMER}.pt"
CKPT_LSTM_NOATTN="${CKPT_DIR}/${RUN_LSTM_NOATTN}.pt"

HIST_LSTM="final/outputs/${RUN_LSTM}/history.json"
HIST_TRANSFORMER="final/outputs/${RUN_TRANSFORMER}/history.json"
HIST_LSTM_NOATTN="final/outputs/${RUN_LSTM_NOATTN}/history.json"

step() {
    echo
    echo "==> $1"
}

train_step() {
    # train.py chỉ ghi checkpoint từ epoch 1 (mỗi lần val loss cải thiện) —
    # nếu chỉ guard theo checkpoint, một run bị ngắt giữa chừng (mất SSH...)
    # sẽ bị coi là đã xong khi chạy lại. history.json chỉ được ghi ở cuối
    # cùng lúc train() kết thúc thành công, nên guard theo CẢ HAI file.
    local ckpt="$1"
    local hist="$2"
    shift 2
    if [ -f "$ckpt" ] && [ -f "$hist" ]; then
        echo "  bỏ qua, đã có checkpoint + history: $ckpt , $hist"
    else
        "$PY" -m final.train "$@"
    fi
}

step "1. Tải dữ liệu Flickr8k"
"$PY" -m final.data.download

step "2. Xây vocab"
"$PY" -m final.data.vocab

step "3. Precompute feature ResNet-50"
"$PY" -m final.data.features

step "4. Train LSTM + attention (run: ${RUN_LSTM})"
train_step "$CKPT_LSTM" "$HIST_LSTM" --decoder lstm $SMOKE_FLAG

step "5. Train Transformer (run: ${RUN_TRANSFORMER})"
train_step "$CKPT_TRANSFORMER" "$HIST_TRANSFORMER" --decoder transformer $SMOKE_FLAG

step "6. Train LSTM không attention — ablation (run: ${RUN_LSTM_NOATTN})"
train_step "$CKPT_LSTM_NOATTN" "$HIST_LSTM_NOATTN" --decoder lstm --no-attention $SMOKE_FLAG

step "7. Evaluate ${RUN_LSTM} (greedy, beam3, beam5)"
"$PY" -m final.evaluate --checkpoint "$CKPT_LSTM" --modes greedy,beam3,beam5 ${EVAL_LIMIT[@]+"${EVAL_LIMIT[@]}"}

step "8. Evaluate ${RUN_TRANSFORMER} (greedy, beam3, beam5)"
"$PY" -m final.evaluate --checkpoint "$CKPT_TRANSFORMER" --modes greedy,beam3,beam5 ${EVAL_LIMIT[@]+"${EVAL_LIMIT[@]}"}

step "9. Evaluate ${RUN_LSTM_NOATTN} (greedy, beam3, beam5)"
"$PY" -m final.evaluate --checkpoint "$CKPT_LSTM_NOATTN" --modes greedy,beam3,beam5 ${EVAL_LIMIT[@]+"${EVAL_LIMIT[@]}"}

step "10. Visualize: learning curves (${RUN_LSTM}, ${RUN_TRANSFORMER}, ${RUN_LSTM_NOATTN})"
"$PY" -m final.visualize --what curves --runs "${RUN_LSTM},${RUN_TRANSFORMER},${RUN_LSTM_NOATTN}"

step "11. Visualize: attention heatmap (${RUN_LSTM})"
"$PY" -m final.visualize --what attention --checkpoint "$CKPT_LSTM" --indices 0,5,17,42

step "12. Visualize: attention heatmap (${RUN_TRANSFORMER})"
"$PY" -m final.visualize --what attention --checkpoint "$CKPT_TRANSFORMER" --indices 0,5,17,42

step "13. Visualize: sample captions theo epoch (${RUN_LSTM}, ${RUN_TRANSFORMER})"
"$PY" -m final.visualize --what samples --runs "${RUN_LSTM},${RUN_TRANSFORMER}"

echo
echo "==> Xong toàn bộ pipeline."
