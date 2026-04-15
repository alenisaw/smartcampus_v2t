# scripts/runtime/smoke_services.py
"""
Service smoke checks for SmartCampus V2T.

Purpose:
- Verify that local API and UI services respond at expected endpoints.
- Provide a lightweight operational check after refactors or environment changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Tuple


def _fetch(url: str, timeout: float) -> Tuple[int, str]:
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return int(resp.status), body


def _check_json_health(url: str, timeout: float) -> bool:
    try:
        status, body = _fetch(url, timeout)
        if status != 200:
            print(f"[FAIL] {url} status={status}")
            return False
        payload = json.loads(body or "{}")
        ok = bool(payload.get("ok", False))
        if not ok:
            print(f"[FAIL] {url} payload={payload}")
            return False
        print(f"[ OK ] {url}")
        return True
    except (urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAIL] {url} error={exc}")
        return False


def _check_status(url: str, timeout: float) -> bool:
    try:
        status, _ = _fetch(url, timeout)
        if status != 200:
            print(f"[FAIL] {url} status={status}")
            return False
        print(f"[ OK ] {url}")
        return True
    except urllib.error.URLError as exc:
        print(f"[FAIL] {url} error={exc}")
        return False


def _check_vllm(url: str, timeout: float) -> bool:
    endpoint = str(url).rstrip("/") + "/v1/models"
    return _check_status(endpoint, timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke checks for API/UI services.")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="Base URL for FastAPI service")
    parser.add_argument("--ui", default="", help="Base URL for Streamlit UI (optional)")
    parser.add_argument("--llm", default="", help="Base URL for vLLM service (optional)")
    parser.add_argument("--timeout", type=float, default=3.0, help="HTTP timeout seconds")
    args = parser.parse_args()

    api = str(args.api).rstrip("/")
    ui = str(args.ui).rstrip("/")
    llm = str(args.llm).rstrip("/")
    timeout = float(args.timeout)

    checks = [
        _check_json_health(f"{api}/healthz", timeout),
        _check_json_health(f"{api}/v1/health", timeout),
        _check_status(f"{api}/v1/index/status", timeout),
        _check_status(f"{api}/v1/videos", timeout),
    ]

    if ui:
        checks.append(_check_status(ui, timeout))
    if llm:
        checks.append(_check_vllm(llm, timeout))

    ok = all(checks)
    if ok:
        print("Smoke checks passed.")
        return 0
    print("Smoke checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
