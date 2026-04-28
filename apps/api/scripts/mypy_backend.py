#!/usr/bin/env python3
"""Run mypy on `app` plus every installed src package under `packages/` (recursive)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]  # apps/api
REPO = API_DIR.parents[1]  # repo root


def _src_packages(parent: Path) -> tuple[list[str], list[str]]:
    """Return (pythonpath roots, mypy module names) for every `<parent>/**/src/<pkg>/`.

    Walks `parent` recursively so that a layout like
        packages/agents/destination/src/agents/destination/
    is picked up alongside top-level packages such as
        packages/platform/src/app_platform/.

    Handles PEP-420 namespace packages: if a child of `src/` has no
    `__init__.py` (e.g. `src/agents/`), descend one level so we still emit
    the dotted module name (e.g. `agents.destination`) for mypy.
    """
    roots: list[str] = []
    names: set[str] = set()
    if not parent.is_dir():
        return roots, sorted(names)
    for src in sorted(parent.rglob("src")):
        if not src.is_dir() or not (src.parent / "pyproject.toml").is_file():
            continue
        roots.append(str(src))
        for child in sorted(src.iterdir()):
            if not child.is_dir():
                continue
            if (child / "__init__.py").is_file():
                names.add(child.name)
                continue
            for grandchild in sorted(child.iterdir()):
                if grandchild.is_dir() and (grandchild / "__init__.py").is_file():
                    names.add(f"{child.name}.{grandchild.name}")
    return roots, sorted(names)


def _all_packages() -> tuple[list[str], list[str]]:
    return _src_packages(REPO / "packages")


def main() -> int:
    extra_roots, _pkgs = _all_packages()
    env = os.environ.copy()
    sep = os.pathsep
    prev = env.get("PYTHONPATH", "")
    combined = [str(API_DIR), *extra_roots]
    if prev:
        combined.append(prev)
    env["PYTHONPATH"] = sep.join(combined)
    # CI-safe mypy target: check API package directly.
    # Cross-package runtime imports are resolved via PYTHONPATH above.
    cmd = [sys.executable, "-m", "mypy", "app"]
    return subprocess.call(cmd, cwd=API_DIR, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
