"""Запуск скриптов организаторов как есть.

Свои RMSE и IoU не считаем: расхождение с их оценщиком дороже собственной формулы.
PYTHONPATH указывает на src их клона, откуда импортируются пакеты task1 и task4.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from trafficflowbench.config_loader import UpstreamCheckout
from trafficflowbench.errors import OrganizerFailed


class OrganizerRunner:
    """Один интерпретатор и один клон на весь прогон базы."""

    def __init__(self, python: Path, upstream: UpstreamCheckout) -> None:
        self.python = python
        self.upstream = upstream

    def run(self, script_name: str, arguments: list[str]) -> None:
        script = self.upstream.script(script_name)
        env = os.environ.copy()
        source = str(self.upstream.checkout / "src")
        previous = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = source if not previous else source + os.pathsep + previous
        env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            [str(self.python), str(script), *arguments],
            cwd=str(self.upstream.checkout),
            env=env,
        )
        if completed.returncode != 0:
            raise OrganizerFailed(f"{script.name} завершился с кодом {completed.returncode}")

    @staticmethod
    def panel_arguments(panels: list[str]) -> list[str]:
        """Повторяющийся --panel в том виде, в каком его читает argparse организаторов."""
        arguments: list[str] = []
        for panel in panels:
            arguments.extend(["--panel", panel])
        return arguments


class ScoreSheet:
    """CSV, который пишет официальный оценщик. Берём строку нужного уровня, свою формулу не повторяем."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise OrganizerFailed(f"оценщик не записал таблицу: {path}")
        self.path = path
        self.frame = pd.read_csv(path)

    def value(self, level: str, column: str, panel: str | None = None) -> float:
        rows = self.frame[self.frame["level"] == level]
        if panel is not None:
            rows = rows[rows["panel"] == panel]
        if rows.empty or column not in self.frame.columns:
            where = f"панель {panel}" if panel is not None else "без отбора панели"
            raise OrganizerFailed(
                f"в {self.path} нет строки level={level} ({where}) с колонкой {column}"
            )
        return float(rows.iloc[0][column])


def current_python() -> Path:
    """Интерпретатор, которым запущен этот процесс. Заглушка WindowsApps сюда не доходит."""
    return Path(sys.executable)
