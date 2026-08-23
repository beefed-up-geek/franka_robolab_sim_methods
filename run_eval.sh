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
    # +Ours — LC 계열만 좌표 조향이 가능한 lang 정책을 쓴다 (vanilla LC 와 동일).
    [LCo_task1]=task1_lang   [LCo_task2]=task2_lang   [LCo_task3]=task3_lang_v10
    [SCo_task1]=task1_abs    [SCo_task2]=task2_abs    [SCo_task3]=task3_abs_v10
    [VLSo_task1]=task1_abs   [VLSo_task2]=task2_abs   [VLSo_task3]=task3_abs_v10
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
    # 로그를 **비운다**. 안 비우면 직전 서버가 남긴 "준비 완료" 를 보고 아직
    # 뜨지도 않은 서버를 다 됐다고 판단해, 러너가 ConnectionRefused 로 즉사한다
    # (실측: 세 조합이 나란히 0에피로 끝났다).
    : > "$REPO/logs/policy_server.log"
    HF_TOKEN="${HF_TOKEN:-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}" \
    nohup ~/hfenv/bin/python "$REPO/inference/policy_server.py" \
        --model ~/_model/"$M"/pretrained_model --port $PORT \
        > "$REPO/logs/policy_server.log" 2>&1 &
    local i; for i in $(seq 1 120); do
        grep -q "준비 완료" "$REPO/logs/policy_server.log" 2>/dev/null && break
        grep -qE "^Traceback|^[A-Za-z]*Error: " "$REPO/logs/policy_server.log" \
            2>/dev/null && {
            echo "✗ 서버 기동 실패"; tail -5 "$REPO/logs/policy_server.log"; return 1; }
        sleep 3
    done
    if ! grep -q "준비 완료" "$REPO/logs/policy_server.log" 2>/dev/null; then
        echo "✗ 서버가 제한시간 안에 준비되지 않았다"
        tail -5 "$REPO/logs/policy_server.log"; return 1
    fi
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
    # 이전 러너가 **완전히 죽을 때까지** 기다린다. SIGTERM 만 보내고 곧바로
    # 새 러너를 띄우면 두 프로세스가 겹치는 순간 ROS 컨텍스트가 깨져
    # "failed to initialize wait set: the given context is not valid" 로 새
    # 러너가 즉사한다 — 지금까지의 간헐적 0에피 실패가 전부 이것이었다.
    docker exec $C pkill -f "methods/.*/.*_runner.py" 2>/dev/null
    for _w in $(seq 1 20); do
        docker exec $C pgrep -f "methods/.*/.*_runner.py" >/dev/null 2>&1 || break
        sleep 1
    done
    docker exec $C pkill -9 -f "methods/.*/.*_runner.py" 2>/dev/null
    sleep 2
    local RUNNER
    case "$M" in
        VLA) RUNNER=VLA/vla_runner.py;;
        LC) RUNNER=LC/vanilla/lc_runner.py;;
        SC) RUNNER=SC/vanilla/sc_runner.py;;
        VLS) RUNNER=VLS/vanilla/vls_runner.py;;
        LCo) RUNNER=LC/ours/lc_ours_runner.py;;
        SCo) RUNNER=SC/ours/sc_ours_runner.py;;
        VLSo) RUNNER=VLS/ours/vls_ours_runner.py;;
    esac
    # task3 은 에피소드가 "세 캔 모두 담기"(2026-08-21 프로토콜)라 3배 길다.
    local TOUT=120
    [ "$T" = "task3" ] && TOUT=300
    local BEFORE=$HAVE
    timeout $((NEED * (TOUT * 2 + 60) + 600)) docker exec $C bash -c \
        "export PYTHONPATH=$B/rclpy:\$PYTHONPATH \
            LD_LIBRARY_PATH=$B/lib:\$LD_LIBRARY_PATH \
            RMW_IMPLEMENTATION=rmw_fastrtps_cpp; \
         /isaac-sim/python.sh /workspace/methods/methods/$RUNNER \
            --task $T --episodes $NEED --timeout $TOUT" 2>&1 \
        | tee "/tmp/raw_${M}_${T}.log" \
        | grep -a --line-buffered -E "^\[(LC|SC|VLS|VLA|LCo|SCo|VLSo|vlm)|Traceback|^[A-Za-z]*(Error|Exception):|No such file" || true
    local AFTER=0
    [ -f "$JS" ] && AFTER=$(wc -l < "$JS")
    if [ "$AFTER" -le "$BEFORE" ]; then
        echo "!! $M/$T: 에피소드가 하나도 안 쌓였다 (${BEFORE}→${AFTER}) — 러너 실패"
    fi
    echo "[eval] $M/$T 종료: ${AFTER}/${EP} $(date +%H:%M)"
}

for M in VLA LC SC VLS LCo SCo VLSo; do
    [ -n "$ONLY_M" ] && [ "$M" != "$ONLY_M" ] && continue
    for T in task1 task2 task3; do
        [ -n "$ONLY_T" ] && [ "$T" != "$ONLY_T" ] && continue
        run_one "$M" "$T"
    done
done
pkill -f "inference/policy_server.py" 2>/dev/null || true
echo "[eval] 스윕 종료 $(date +%m/%d\ %H:%M)"
