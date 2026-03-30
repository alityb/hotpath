from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    package_root = Path(__file__).resolve().parent
    binary = package_root / "bin" / "rlprof"
    if not binary.exists():
        raise SystemExit(f"missing rlprof binary at {binary}")

    env = os.environ.copy()
    env.setdefault("RLPROF_PYTHON_EXECUTABLE", sys.executable)
    os.execve(str(binary), [str(binary), *sys.argv[1:]], env)
