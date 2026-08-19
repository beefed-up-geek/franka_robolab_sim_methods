# SPDX-License-Identifier: Apache-2.0
"""OpenRouter VLM 클라이언트 — 에피소드당 호출 예산이 핵심이다.

사용자 제약: task 1회 수행(에피소드)에 VLM 호출 30번 이하. 예산은 이 클래스가
강제한다 — 소진되면 call() 이 None 을 돌려주고, 호출부는 마지막 지시/보상을
그대로 유지한 채 계속 간다 (정책이 멈추는 것보다 낫다).

gemini-3.7-flash 는 **추론(thinking) 모델**이고 reasoning 비활성화가 불가다
("Reasoning is mandatory", 실측 400). effort=low 로 제한하면 사고를 거의 안
하고 1.4초에 답한다 — 그래도 content 가 비면 예산 4배로 한 번 재시도한다.
"""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request

ENV_PATH = "/workspace/methods/.env"
MODEL = "google/gemini-3.7-flash"
URL = "https://openrouter.ai/api/v1/chat/completions"


def _load_key(path: str = ENV_PATH) -> str:
    for line in open(path):
        m = re.match(r"\s*OPENROUTER_API_KEY\s*=\s*(\S+)", line)
        if m:
            return m.group(1)
    raise RuntimeError(f"OPENROUTER_API_KEY 가 없습니다: {path}")


class VLM:
    def __init__(self, budget: int = 30, model: str = MODEL,
                 log_path: str | None = None) -> None:
        self.key = _load_key()
        self.model = model
        self.budget = budget
        self.used = 0          # 이번 에피소드 사용량
        self.total = 0         # 전체 사용량 (러너 수명)
        self.log_path = log_path

    def reset_budget(self) -> None:
        self.used = 0

    # ── 호출 ──────────────────────────────────────────────────────────
    def call(self, prompt: str, images: list[bytes] = (), system: str | None = None,
             max_tokens: int = 900, timeout: float = 45.0) -> str | None:
        """텍스트 응답 하나. 예산 소진·실패 시 None (호출부가 직전 값 유지)."""
        if self.used >= self.budget:
            return None
        content: list[dict] = [{"type": "text", "text": prompt}]
        for jpg in images:
            b64 = base64.b64encode(jpg).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        messages = ([{"role": "system", "content": system}] if system else []) \
            + [{"role": "user", "content": content}]

        for attempt, mt in enumerate((max_tokens, max_tokens * 4)):
            body = {"model": self.model, "messages": messages, "max_tokens": mt,
                    "reasoning": {"effort": "low"}}
            try:
                req = urllib.request.Request(
                    URL, data=json.dumps(body).encode(),
                    headers={"Authorization": f"Bearer {self.key}",
                             "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    d = json.loads(r.read())
                text = (d.get("choices") or [{}])[0].get("message", {}).get("content")
                self.used += 1
                self.total += 1
                if self.log_path:
                    with open(self.log_path, "a") as f:
                        f.write(json.dumps({
                            "t": time.time(), "used": self.used,
                            "tokens": d.get("usage", {}).get("total_tokens"),
                            "prompt": prompt[:120],
                            "reply": (text or "")[:200]}, ensure_ascii=False) + "\n")
                if text:               # 추론이 다 먹었으면(빈 응답) 재시도
                    return text
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt == 1:
                    print(f"[vlm] 호출 실패: {e}", flush=True)
                time.sleep(1.5)
        return None

    def call_json(self, prompt: str, **kw) -> dict | None:
        """JSON 오브젝트 응답. 코드펜스·잡담 속에서도 첫 {} 블록을 건진다."""
        text = self.call(prompt + "\n\nRespond with a single JSON object only.", **kw)
        if not text:
            return None
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
