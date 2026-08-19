# LC — Steerable Policies (arXiv:2602.13193) vanilla 구현

VLM 상위 지휘(supervision) 구성 — 논문 Fig.4(b) 의 in-context reasoning VLM.

## 논문 → 구현 대응

| 논문 구성요소 | 구현 (`lc_runner.py`) | 충실도 |
|---|---|---|
| Steerable Policy — 4계층 steering command 로 학습한 저수준 VLA | 우리 task*_lang 모델 (task/subtask/atomic motion/월드좌표 point 로 학습, `scripts/host/ingest_lang.py`) | **충실** — 논문은 픽셀 좌표, 우리는 정책 상태공간과 같은 월드 좌표 |
| off-the-shelf VLM 이 관측·이력에서 명령 추상화를 **골라** 발행 | gemini-3.7-flash 에 front+wrist 카메라, 물체 좌표, 명령 스타일 안내, 직전 명령 이력을 주고 JSON {reasoning, command} 수신 | **충실** |
| N 스텝마다 재지휘 (논문 N=20) | **동기 사이클** — VLM 호출(로봇 정지 대기) → 명령으로 청크 1개(16스텝) 생성 → 실행 → 반복. 급변 이벤트(팔 진입·안전·라운드)는 남은 청크를 버려 즉시 재지휘. 사이클 ≈ 4초(VLM) + 2.7초(실행) → ~18콜/120초, 예산 30 안 | **충실** (지휘·실행 엄격 교대판) |
| 호출 예산 | 에피소드당 ≤30 (사용자 제약). 소진 시 마지막 명령 유지 | 추가 제약 |

## 실행

```
./run_eval.sh LC task3 5      # 스모크
./run_eval.sh LC              # 3태스크 × 50에피
```
