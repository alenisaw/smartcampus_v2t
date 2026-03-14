# scripts/check_ui_sanity.py
"""Sanity checks for the Streamlit UI layer.

Purpose:
- Catch source-level mojibake before it reaches the demo UI.
- Detect duplicate top-level function definitions in frontend modules.
- Verify that `app.main` and the page-oriented UI modules still import cleanly.
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

PYTHON_FILES = [
    ROOT / "app" / "main.py",
    ROOT / "app" / "state.py",
    ROOT / "app" / "api_client.py",
    *sorted((ROOT / "app" / "view").glob("*.py")),
]

TEXT_FILES = [
    ROOT / "app" / "main.py",
    ROOT / "app" / "state.py",
    ROOT / "app" / "assets" / "ui_text.json",
    *sorted((ROOT / "app" / "view").glob("*.py")),
]

STYLES_PATH = ROOT / "app" / "assets" / "styles.css"

MOJIBAKE_PATTERNS = (
    "РћС",
    "РџРѕР",
    "РћС‚С‡",
    "РЈРјРЅС‹Р№",
    "ТљР°С‚Рµ",
    "Р†Р·РґРµСѓ",
    "вЊ«",
    "в†»",
    "âŒ«",
    "â†»",
)

MODULE_IMPORTS = (
    "app.main",
    "app.view.shared",
    "app.view.storage",
    "app.view.analytics",
    "app.view.search",
    "app.view.reports",
    "app.view.overview",
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


def _check_mojibake() -> list[str]:
    errors: list[str] = []
    for path in TEXT_FILES:
        text = path.read_text(encoding="utf-8")
        hits = [pattern for pattern in MOJIBAKE_PATTERNS if pattern in text]
        if hits:
            errors.append(f"mojibake markers in {path.relative_to(ROOT)}: {', '.join(hits)}")
    return errors


def _check_imports() -> list[str]:
    errors: list[str] = []
    for module_name in MODULE_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - smoke path
            errors.append(f"import failed: {module_name}: {exc}")
    return errors


def _check_legacy_ui_refs() -> list[str]:
    errors: list[str] = []
    forbidden = ("from app import ui", "import app.ui", "from app.ui")
    scan_paths = [
        ROOT / "app" / "main.py",
        *sorted((ROOT / "app" / "view").glob("*.py")),
    ]
    for path in scan_paths:
        text = path.read_text(encoding="utf-8")
        hits = [pattern for pattern in forbidden if pattern in text]
        if hits:
            errors.append(f"legacy app.ui reference in {path.relative_to(ROOT)}: {', '.join(hits)}")
    return errors


def _check_markers() -> list[str]:
    errors: list[str] = []
    styles_text = STYLES_PATH.read_text(encoding="utf-8")
    css_tokens = set()
    for line in styles_text.splitlines():
        for token in line.replace(",", " ").split():
            if token.startswith("."):
                css_tokens.add(token[1:])

    code_markers: set[str] = set()
    for path in PYTHON_FILES:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "_mark":
                continue
            for arg in node.args:
                if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                    continue
                for token in arg.value.split():
                    if token.endswith("-marker"):
                        code_markers.add(token)

    missing = sorted(marker for marker in code_markers if marker not in css_tokens)
    if missing:
        errors.append(f"missing css marker definitions: {', '.join(missing)}")
    return errors


def main() -> int:
    errors: list[str] = []
    errors.extend(_check_compile())
    errors.extend(_check_duplicate_defs())
    errors.extend(_check_mojibake())
    errors.extend(_check_markers())
    errors.extend(_check_legacy_ui_refs())
    errors.extend(_check_imports())

    if errors:
        for line in errors:
            print(f"[ui-sanity] {line}")
        return 1

    print("[ui-sanity] ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
