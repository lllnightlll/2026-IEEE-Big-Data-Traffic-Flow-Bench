"""Паспорт коридора: порядок связей и границы фундаментальной диаграммы.

Порядок берётся из топологии организатора. Проекция не даёт скорости выше свободной,
потоку — выше вместимости, а плотности q/v — выше плотности затора.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trafficflowbench.errors import ConfigError


@dataclass(frozen=True)
class LinkChain:
    """Главные связи коридора в порядке движения, без разрывов нумерации."""

    link_ids: tuple[str, ...]
    position: dict[str, int]

    @classmethod
    def from_topology(cls, frame: pd.DataFrame) -> LinkChain:
        if "link_id" not in frame.columns or "order_index" not in frame.columns:
            raise ConfigError("топология коридора должна содержать link_id и order_index")
        ordered = frame[["link_id", "order_index"]].copy()
        ordered["link_id"] = ordered["link_id"].astype(str)
        ordered["order_index"] = pd.to_numeric(ordered["order_index"], errors="coerce")
        ordered = ordered.dropna(subset=["order_index"]).drop_duplicates("link_id")
        ordered = ordered.sort_values("order_index", kind="mergesort")
        link_ids = tuple(ordered["link_id"].tolist())
        return cls(link_ids=link_ids, position={link_id: index for index, link_id in enumerate(link_ids)})

    def upstream(self, link_id: str) -> str | None:
        """Сосед против движения. Очередь распространяется туда, а не вниз по потоку.

        order_index в топологии растёт по ходу движения: у первой связи входящих
        главных связей нет, у следующей во входящих стоит предыдущая.
        """
        index = self.position.get(str(link_id))
        if index is None or index == 0:
            return None
        return self.link_ids[index - 1]

    @classmethod
    def read(cls, panel_dir: Path) -> LinkChain:
        path = panel_dir / "network" / "lwr_mainline_topology.csv"
        if not path.is_file():
            raise ConfigError(f"нет топологии коридора: {path}")
        return cls.from_topology(pd.read_csv(path))


@dataclass(frozen=True)
class DiagramBounds:
    """Ограничения связи. Неизвестная связь получает максимум по коридору, а не выдуманное число."""

    free_speed_kmh: dict[str, float]
    capacity_vph: dict[str, float]
    jam_density_per_km: dict[str, float]
    speed_epsilon_kmh: float

    @classmethod
    def from_parameters(cls, frame: pd.DataFrame, speed_epsilon_kmh: float) -> DiagramBounds:
        required = {"link_id", "free_speed_kmh", "capacity_vph", "k_jam"}
        missing = required - set(frame.columns)
        if missing:
            raise ConfigError(f"в параметрах диаграммы нет колонок {sorted(missing)}")
        table = frame.copy()
        table["link_id"] = table["link_id"].astype(str)
        return cls(
            free_speed_kmh=_positive_map(table, "free_speed_kmh"),
            capacity_vph=_positive_map(table, "capacity_vph"),
            jam_density_per_km=_positive_map(table, "k_jam"),
            speed_epsilon_kmh=speed_epsilon_kmh,
        )

    @classmethod
    def read(cls, panel_dir: Path, speed_epsilon_kmh: float) -> DiagramBounds:
        path = panel_dir / "network" / "fd_parameters.csv"
        if not path.is_file():
            raise ConfigError(f"нет параметров диаграммы: {path}")
        return cls.from_parameters(pd.read_csv(path), speed_epsilon_kmh)

    def clip(self, link_ids: np.ndarray, speed: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Вернуть пару, которая может лежать на диаграмме этой связи.

        Плотность здесь q/v в авт/км, в тех же единицах, что и k_jam файла параметров.
        Сохранение автомобилей между связями этот шаг не проверяет: это следующий слой.
        """
        free = self._column(link_ids, self.free_speed_kmh)
        capacity = self._column(link_ids, self.capacity_vph)
        jam = self._column(link_ids, self.jam_density_per_km)
        clipped_speed = np.clip(speed, 0.0, free)
        clipped_flow = np.clip(flow, 0.0, capacity)
        density = np.divide(
            clipped_flow,
            np.maximum(clipped_speed, self.speed_epsilon_kmh),
        )
        # При почти нулевой скорости отношение q/v не является плотностью. Поток там уже ограничен вместимостью.
        moving = clipped_speed >= self.speed_epsilon_kmh
        too_dense = moving & np.isfinite(jam) & (density > jam)
        clipped_flow = np.where(too_dense, np.minimum(jam * clipped_speed, capacity), clipped_flow)
        return clipped_speed, np.clip(clipped_flow, 0.0, capacity)

    @staticmethod
    def _column(link_ids: np.ndarray, values: dict[str, float]) -> np.ndarray:
        if not values:
            return np.full(len(link_ids), np.inf, dtype=float)
        default = max(values.values())
        mapped = pd.Series(link_ids).map(values)
        return mapped.fillna(default).to_numpy(dtype=float)


def _positive_map(frame: pd.DataFrame, column: str) -> dict[str, float]:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    valid = numeric.notna() & (numeric > 0)
    return dict(zip(frame.loc[valid, "link_id"].astype(str), numeric[valid].astype(float)))
