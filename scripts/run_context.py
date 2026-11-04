"""Точка входа слоя State v2."""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from trafficflowbench.cli.run_context import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
