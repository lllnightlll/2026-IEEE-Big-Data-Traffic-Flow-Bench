"""Сборка файла очереди по окнам релиза. Будущее относительно T не читается."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trafficflowbench.errors import ProjectError
from trafficflowbench.queue.forecast import (
    QueueHorizon,
    RecurrentQueueProfile,
    latest_speeds,
    speed_thresholds,
)
from trafficflowbench.queue.settings import QueueSettings
from trafficflowbench.state.corridor import LinkChain

HISTORY_COLUMNS = ["window_id", "timestamp", "link_id", "speed_kmh", "is_score_eligible"]
ORIGIN_COLUMNS = ["timestamp", "link_id", "speed_kmh", "is_score_eligible"]


class QueuePanelBuilder:
    """Один коридор: пороги и повторяемость train один раз, затем оба сплита сдачи."""

    def __init__(self, panel: str, release_root: Path, settings: QueueSettings, free_speed_fraction: float, step_minutes: int) -> None:
        self.panel = panel
        self.release_root = release_root
        self.settings = settings
        self.step_minutes = step_minutes
        self.panel_dir = release_root / "corridors" / panel
        self.chain = LinkChain.read(self.panel_dir)
        self.thresholds = speed_thresholds(self.panel_dir, free_speed_fraction)
        self.horizon = QueueHorizon(settings)
        self.recurrent = RecurrentQueueProfile.fit(self.panel_dir, self.chain.link_ids, self.thresholds, step_minutes)

    def predict_split(self, split: str) -> pd.DataFrame:
        """Строки шаблона этого сплита. Метки времени остаются байтами шаблона."""
        task_dir = self.release_root / "task2" / self.panel / split
        windows = self._read_windows(task_dir / "window_index.csv")
        history = pd.read_parquet(task_dir / "window_history.parquet", columns=HISTORY_COLUMNS)
        history["window_id"] = history["window_id"].astype(str)
        template = pd.read_csv(task_dir / "sample_submission_queue.csv", dtype=str)
        if "queue_pred" in template.columns:
            template = template.drop(columns=["queue_pred"])
        template["window_id"] = template["window_id"].astype(str)
        origins = self._origins(split, set(windows["origin"]))
        pieces: list[pd.DataFrame] = []
        for window_id, part in template.groupby("window_id", sort=False):
            if window_id not in windows.index:
                raise ProjectError(f"окно {window_id} есть в шаблоне и нет в индексе {self.panel}/{split}")
            info = windows.loc[window_id]
            observed = history[history["window_id"] == window_id]
            at_origin = origins.get(info["origin"])
            if at_origin is not None and not at_origin.empty:
                observed = pd.concat([observed, at_origin], ignore_index=True)
            speeds = latest_speeds(observed, info["origin"])
            pieces.append(
                self.horizon.predict(
                    str(info["condition"]),
                    part,
                    speeds,
                    self.chain,
                    self.thresholds,
                    self.recurrent,
                )
            )
        if not pieces:
            raise ProjectError(f"шаблон очереди пуст: {self.panel}/{split}")
        return pd.concat(pieces, ignore_index=True)

    def _read_windows(self, path: Path) -> pd.DataFrame:
        if not path.is_file():
            raise ProjectError(f"нет индекса окон: {path}")
        frame = pd.read_csv(path, dtype=str)
        required = {"window_id", "forecast_origin", "condition"}
        missing = required - set(frame.columns)
        if missing:
            raise ProjectError(f"{path} не содержит {sorted(missing)}")
        frame["window_id"] = frame["window_id"].astype(str)
        frame["origin"] = pd.to_datetime(frame["forecast_origin"], utc=True)
        if frame["origin"].isna().any():
            raise ProjectError(f"в {path} есть неразборчивый forecast_origin")
        duplicated = int(frame["window_id"].duplicated().sum())
        if duplicated:
            raise ProjectError(f"{path} содержит {duplicated} повторов window_id")
        return frame.set_index("window_id")

    def _origins(self, split: str, needed: set[pd.Timestamp]) -> dict[pd.Timestamp, pd.DataFrame]:
        """Строки ровно на T из маскированного слоя. Горизонт окна сюда не входит."""
        directory = self.panel_dir / split / "mainline_states_masked"
        files = sorted(directory.glob("**/*.parquet")) if directory.is_dir() else []
        pieces: list[pd.DataFrame] = []
        for path in files:
            frame = pd.read_parquet(path, columns=ORIGIN_COLUMNS)
            times = pd.to_datetime(frame["timestamp"], utc=True)
            keep = times.isin(needed)
            if not bool(keep.any()):
                continue
            piece = frame.loc[keep, ["timestamp", "link_id", "speed_kmh", "is_score_eligible"]].copy()
            piece["timestamp"] = piece["timestamp"].astype(str)
            pieces.append(piece)
        if not pieces:
            return {}
        stacked = pd.concat(pieces, ignore_index=True)
        stacked["_origin"] = pd.to_datetime(stacked["timestamp"], utc=True)
        return {
            origin: group.drop(columns=["_origin"])
            for origin, group in stacked.groupby("_origin", sort=False)
        }
