"""
Small HTTP healthcheck helper used by Docker health probes.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="HTTP healthcheck probe.")
    parser.add_argument("url", help="Probe URL")
    parser.add_argument("--timeout", type=float, default=3.0, help="HTTP timeout in seconds")
    parser.add_argument(
        "--expect-json-ok",
        action="store_true",
        help="Require a JSON object with ok=true",
    )
    args = parser.parse_args()

    req = urllib.request.Request(url=str(args.url), method="GET")
    with urllib.request.urlopen(req, timeout=float(args.timeout)) as response:
        if int(response.status) != 200:
            return 1
        if not args.expect_json_ok:
            return 0
        payload = json.loads(response.read().decode("utf-8", errors="replace") or "{}")
        return 0 if bool(payload.get("ok", False)) else 1


if __name__ == "__main__":
    sys.exit(main())
