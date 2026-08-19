# VLS — Vision-Language Steering (arXiv:2602.03973) vanilla 구현

학습 없는 추론 시점 조향 — VLM 이 합성한 미분 가능 보상으로 flow-matching
정책(GR00T N1.7 abs)의 샘플링을 유도한다.

## 논문 → 구현 대응

| 논문 구성요소 | 구현 (`vls_runner.py` + policy_server `/act_chunk`) | 충실도 |
|---|---|---|
| OOD 접지: SAM/DINOv2/깊이 → 3D 키포인트 스캐폴드 P | 시뮬 특권 상태의 물체·통·홈·단자 좌표를 `kp` 딕셔너리로 제공 | **대체** — 지각 스택 불필요, 스캐폴드의 역할(보상의 공간 변수)은 동일 |
| VLM 이 태스크를 스테이지로 분해, 스테이지별 **프로그램 보상** R_s(traj) 합성 (torch 연산, off-graph VLM) | gemini-3.7-flash 가 스테이지별 `reward(traj, kp)` 파이썬 본문 생성 → torch·math 만 보이는 이름공간에 exec. 실패 시 재시도 ≤2 | **충실** |
| ∇R 을 디노이징 스텝에 주입 + 입자 리샘플링 | `/act_chunk` 가 flow 헤드에서 K=6 후보 청크 샘플(입자) → 각 후보에 ∇R 경사상승 5스텝(±4cm 클램프) → R 최대 후보 실행 | **부분 대체** — 유도를 내부 디노이징 스텝이 아니라 완성 표본에 가함 (논문의 gradient-free resampling + gradient refinement 조합). 정책 내부 수정 없음 = plug-and-play 유지 |
| 폐루프 스테이지 전환 | VLM 이 스테이지별 `done(kp, eef, gripper)` 술어 동봉, 러너가 매 스텝 평가 | **충실** |
| 청크 일부만 실행 후 재계획 | 16스텝 청크 중 8스텝 실행 후 재샘플 | **충실** |

## 실행

```
./run_eval.sh VLS task3 5     # 스모크
./run_eval.sh VLS             # 3태스크 × 50에피
```

VLM 호출: 에피소드당 계획 1회(+재시도 ≤2, task3 라운드 재배치 시 재계획).
