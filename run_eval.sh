#!/usr/bin/env bash
# 산업 실험 스윕 — (방법 × 태스크) 조합을 순차 실행한다.
#
#   ./run_eval.sh                 # 9조합 × 50에피 전부
#   ./run_eval.sh SC task2 5      # 한 조합만 (스모크)
#
# 조합마다: 시뮬 환경 확인·교체 → 정책 서버 재기동(해당 모델) → 러너 실행.
# 결과는 results/raw/{방법}_{태스크}.jsonl 에 에피소드 단위로 쌓인다 —
# 중단 후 재실행하면 이미 채운 조합은 목표 수를 세어 건너뛴다.
set -uo pipefail
ONLY_M="${1:-}"; ONLY_T="${2:-}"; EP="${3:-50}"
REPO=~/franka_robolab_sim
MREPO=~/franka_robolab_sim_methods
RAW=$MREPO/results/raw
mkdir -p "$RAW"
C=franka_robolab_sim
B=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy
PORT=8010

declare -A MODEL_OF=(
    [VLA_task1]=task1_abs    [VLA_task2]=task2_abs    [VLA_task3]=task3_abs_v10
    [LC_task1]=task1_lang    [LC_task2]=task2_lang    [LC_task3]=task3_lang_v10
    [SC_task1]=task1_abs     [SC_task2]=task2_abs     [SC_task3]=task3_abs_v10
    [VLS_task1]=task1_abs    [VLS_task2]=task2_abs    [VLS_task3]=task3_abs_v10
    [VLSa_task1]=task1_abs   [VLSa_task2]=task2_abs   [VLSa_task3]=task3_abs_v10
)
declare -A ENV_OF=( [task1]=task1 [task2]=task2_test [task3]=task3_test )

sim_env() {     # demo_all.sh 의 검증된 전환 루틴
    local WANT="$1" CUR
    CUR=$(docker exec $C pgrep -af "env/script/run.py" 2>/dev/null \
          | grep -oE "task[123](_train|_test)?" | head -1)
    [ "$CUR" = "$WANT" ] && return 0
    echo "[eval] 환경 전환 → $WANT (기동 2~5분)"
    cd "$REPO" && ./scripts/sim_stop.sh >/dev/null 2>&1
    local w; for w in $(seq 1 30); do
        docker exec $C pgrep -f "env/script/run.py" >/dev/null 2>&1 || break
        sleep 2
    done
    sleep 2
    ./scripts/sim_start.sh "$WANT" >/dev/null 2>&1
    local i; for i in $(seq 1 150); do
        curl -s --max-time 3 http://127.0.0.1:8003/telemetry 2>/dev/null \
            | grep -qE '"ee_x": *-?[0-9]' && break
        sleep 5
    done
    sleep 5
}

serve() {       # run.sh 의 서버 기동 루틴 (모델 교체 시에만 재기동)
    local M="$1"
    if [ -f /tmp/eval_server.model ] && [ "$(cat /tmp/eval_server.model)" = "$M" ] \
       && pgrep -f "inference/policy_server.py" >/dev/null; then
        return 0
    fi
    pkill -f "inference/policy_server.py" 2>/dev/null || true
    sleep 2
    HF_TOKEN="${HF_TOKEN:-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}" \
    nohup ~/hfenv/bin/python "$REPO/inference/policy_server.py" \
        --model ~/_model/"$M"/pretrained_model --port $PORT \
        > "$REPO/logs/policy_server.log" 2>&1 &
    local i; for i in $(seq 1 120); do
        grep -q "준비 완료" "$REPO/logs/policy_server.log" 2>/dev/null && break
        grep -qE "Traceback|Error" "$REPO/logs/policy_server.log" 2>/dev/null && {
            echo "✗ 서버 기동 실패"; tail -5 "$REPO/logs/policy_server.log"; return 1; }
        sleep 3
    done
    echo "$M" > /tmp/eval_server.model
}

run_one() {
    local M="$1" T="$2"
    local JS="$RAW/${M}_${T}.jsonl"
    local HAVE=0
    [ -f "$JS" ] && HAVE=$(wc -l < "$JS")
    if [ "$HAVE" -ge "$EP" ]; then
        echo "[eval] $M/$T: 이미 ${HAVE}에피 — 건너뜀"; return 0
    fi
    local NEED=$((EP - HAVE))
    echo "===== $M/$T — ${NEED}에피 (누적 ${HAVE}/${EP}) $(date +%H:%M) ====="
    sim_env "${ENV_OF[$T]}" || return 1
    serve "${MODEL_OF[${M}_${T}]}" || return 1
    docker exec $C pkill -f "methods/.*/.*_runner.py" 2>/dev/null
    local RUNNER
    case "$M" in
        VLA) RUNNER=VLA/vla_runner.py;;
        LC) RUNNER=LC/vanilla/lc_runner.py;;
        SC) RUNNER=SC/vanilla/sc_runner.py;;
        VLS) RUNNER=VLS/vanilla/vls_runner.py;;
        VLSa) RUNNER=VLS_authentic/vls_authentic_runner.py;;
    esac
    # task3 은 에피소드가 "세 캔 모두 담기"(2026-08-21 프로토콜)라 3배 길다.
    local TOUT=120
    [ "$T" = "task3" ] && TOUT=300
    timeout $((NEED * (TOUT * 2 + 60) + 600)) docker exec $C bash -c \
        "export PYTHONPATH=$B/rclpy:\$PYTHONPATH \
            LD_LIBRARY_PATH=$B/lib:\$LD_LIBRARY_PATH \
            RMW_IMPLEMENTATION=rmw_fastrtps_cpp; \
         /isaac-sim/python.sh /workspace/methods/methods/$RUNNER \
            --task $T --episodes $NEED --timeout $TOUT" 2>&1 \
        | grep -aE "^\[(LC|SC|VLS|VLSa|VLA|vlm)" || true
}

for M in VLA LC SC VLS VLSa; do
    [ -n "$ONLY_M" ] && [ "$M" != "$ONLY_M" ] && continue
    for T in task1 task2 task3; do
        [ -n "$ONLY_T" ] && [ "$T" != "$ONLY_T" ] && continue
        run_one "$M" "$T"
    done
done
pkill -f "inference/policy_server.py" 2>/dev/null || true
echo "[eval] 스윕 종료 $(date +%m/%d\ %H:%M)"
