# scripts/check_backend_sanity.py
"""
Sanity checks for the backend runtime layer.

Purpose:
- Catch broken imports and duplicate top-level definitions before launching backend services.
- Detect stale references to removed backend modules and outdated UI entrypoints.
"""

from __future__ import annotations

import ast
import importlib
import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON_FILES = sorted((ROOT / "backend").rglob("*.py"))
RUN_ALL_PATH = ROOT / "run_all.bat"

MODULE_IMPORTS = (
    "backend.api",
    "backend.http.common",
    "backend.http.grounded",
    "backend.jobs.control",
    "backend.jobs.queue_runtime",
    "backend.jobs.store",
    "backend.jobs.runtime_common",
    "backend.jobs.translate_runtime",
    "backend.jobs.process_runtime",
    "backend.jobs.worker_runtime",
    "backend.worker",
)


def _check_compile() -> list[str]:
    errors: list[str] = []
    for path in PYTHON_FILES:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"compile failed: {path.relative_to(ROOT)}: {exc.msg}")
    return errors


def _check_duplicate_defs() -> list[str]:
    errors: list[str] = []
    for path in PYTHON_FILES:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        seen: dict[str, int] = {}
        duplicates: list[str] = []
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name in seen:
                duplicates.append(node.name)
            seen[node.name] = node.lineno
        if duplicates:
            names = ", ".join(sorted(set(duplicates)))
            errors.append(f"duplicate defs: {path.relative_to(ROOT)}: {names}")
    return errors


def _check_imports() -> list[str]:
    errors: list[str] = []
    for module_name in MODULE_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - smoke path
            errors.append(f"import failed: {module_name}: {exc}")
    return errors


def _check_legacy_refs() -> list[str]:
    errors: list[str] = []
    forbidden = ("backend.job_executors", "app.ui", "app\\ui.py")
    scan_paths = [*PYTHON_FILES, RUN_ALL_PATH]
    for path in scan_paths:
        text = path.read_text(encoding="utf-8")
        hits = [pattern for pattern in forbidden if pattern in text]
        if hits:
            errors.append(f"legacy reference in {path.relative_to(ROOT)}: {', '.join(hits)}")
    return errors


def _check_run_all() -> list[str]:
    errors: list[str] = []
    text = RUN_ALL_PATH.read_text(encoding="utf-8")
    required_tokens = ("backend.api:app", "backend.worker", "app\\main.py")
    missing = [token for token in required_tokens if token not in text]
    if missing:
        errors.append(f"run_all missing required tokens: {', '.join(missing)}")
    return errors


def main() -> int:
    errors: list[str] = []
    errors.extend(_check_compile())
    errors.extend(_check_duplicate_defs())
    errors.extend(_check_legacy_refs())
    errors.extend(_check_run_all())
    errors.extend(_check_imports())

    if errors:
        for line in errors:
            print(f"[backend-sanity] {line}")
        return 1

    print("[backend-sanity] ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
