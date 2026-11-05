"""Очередь на горизонте T+5…T+30 без чтения будущего.

Ongoing начинается с последней пригодной скорости на отрезке [T-60, T].
Хвост этой очереди за горизонт сдвигается на соседа выше по потоку.
Onset в этом отрезке ещё не очередь: связь включается только если в train
в тот же день недели и на той же пятиминутке скорость была не выше порога
чаще, чем нет. Метки окон скрыты, поэтому повторяемость считается по
определению очереди на опубликованном train, а не по файлу меток.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trafficflowbench.errors import ProjectError
from trafficflowbench.queue.settings import QueueSettings
from trafficflowbench.state.corridor import LinkChain


def latest_speeds(frame: pd.DataFrame, origin: str | pd.Timestamp) -> dict[str, float]:
    """Последняя пригодная скорость связи не позже T.

    Строка горизонта и час после него в этот словарь не попадают, даже если
    в маскированном слое на их месте лежит пустая или чужая ячейка.
    Непригодная строка ровно на T не затирает показание T-5.
    """
    if frame.empty:
        return {}
    required = {"timestamp", "link_id", "speed_kmh", "is_score_eligible"}
    missing = required - set(frame.columns)
    if missing:
        raise ProjectError(f"в истории очереди нет колонок {sorted(missing)}")
    origin_ts = pd.to_datetime(origin, utc=True)
    times = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    speed = pd.to_numeric(frame["speed_kmh"], errors="coerce")
    eligible = pd.to_numeric(frame["is_score_eligible"], errors="coerce").fillna(0.0) >= 0.5
    keep = (times <= origin_ts) & speed.notna() & eligible
    if not bool(keep.any()):
        return {}
    kept = pd.DataFrame(
        {
            "timestamp": times[keep].to_numpy(),
            "link_id": frame.loc[keep, "link_id"].astype(str).to_numpy(),
            "speed_kmh": speed[keep].to_numpy(dtype=float),
        }
    )
    # Несколько станций одной связи: очередь есть, если хотя бы одна уже не быстрее порога.
    # Для снимка хранится минимум пригодных скоростей в этот момент, его и сравнивают с порогом.
    kept = kept.groupby(["link_id", "timestamp"], as_index=False, sort=False)["speed_kmh"].min()
    latest = kept.sort_values("timestamp", kind="mergesort").groupby("link_id", sort=False).tail(1)
    return dict(zip(latest["link_id"].astype(str), latest["speed_kmh"].astype(float)))


class RecurrentQueueProfile:
    """Доля дней train, когда связь в эти сутки и эту пятиминутку уже была очередью."""

    def __init__(self, link_ids: tuple[str, ...], step_minutes: int) -> None:
        if step_minutes <= 0 or (24 * 60) % step_minutes != 0:
            raise ProjectError("шаг очереди должен делить сутки без остатка")
        self.position = {link_id: index for index, link_id in enumerate(link_ids)}
        self.step_minutes = step_minutes
        self.slots = (24 * 60) // step_minutes
        width = len(link_ids)
        self.seen = np.zeros((width, 7, self.slots), dtype=np.int16)
        self.queued = np.zeros((width, 7, self.slots), dtype=np.int16)

    def update(self, frame: pd.DataFrame, thresholds: dict[str, float]) -> None:
        """Один день train. Повтор той же связи в ту же пятиминутку считается одним днём."""
        if frame.empty:
            return
        times = pd.to_datetime(frame["timestamp"], utc=True)
        link_ids = frame["link_id"].astype(str)
        speed = pd.to_numeric(frame["speed_kmh"], errors="coerce")
        known = link_ids.map(self.position)
        valid = speed.notna() & known.notna() & times.notna()
        if not bool(valid.any()):
            return
        minute = times.dt.hour.to_numpy() * 60 + times.dt.minute.to_numpy()
        slot = (minute // self.step_minutes).astype(np.int64)
        threshold = link_ids.map(thresholds)
        queued = speed.to_numpy(dtype=float) <= threshold.to_numpy(dtype=float)
        table = pd.DataFrame(
            {
                "position": known.to_numpy()[valid.to_numpy()].astype(np.int64),
                "weekday": times.dt.weekday.to_numpy()[valid.to_numpy()].astype(np.int64),
                "slot": slot[valid.to_numpy()],
                "queued": queued[valid.to_numpy()],
            }
        )
        collapsed = table.groupby(["position", "weekday", "slot"], as_index=False, sort=False)["queued"].any()
        positions = collapsed["position"].to_numpy(dtype=np.int64)
        weekdays = collapsed["weekday"].to_numpy(dtype=np.int64)
        slots = collapsed["slot"].to_numpy(dtype=np.int64)
        in_day = (slots >= 0) & (slots < self.slots)
        np.add.at(self.seen, (positions[in_day], weekdays[in_day], slots[in_day]), 1)
        flagged = collapsed["queued"].to_numpy(dtype=bool) & in_day
        np.add.at(self.queued, (positions[flagged], weekdays[flagged], slots[flagged]), 1)

    def share(self, link_id: str, timestamp: pd.Timestamp) -> float | None:
        """None, если этой пятиминутки в train не было. Иначе доля дней с очередью."""
        index = self.position.get(str(link_id))
        if index is None or pd.isna(timestamp):
            return None
        slot = (int(timestamp.hour) * 60 + int(timestamp.minute)) // self.step_minutes
        if not 0 <= slot < self.slots:
            return None
        seen = int(self.seen[index, int(timestamp.weekday()), slot])
        if seen <= 0:
            return None
        return float(self.queued[index, int(timestamp.weekday()), slot]) / float(seen)

    @classmethod
    def fit(
        cls,
        panel_dir: Path,
        link_ids: tuple[str, ...],
        thresholds: dict[str, float],
        step_minutes: int,
    ) -> RecurrentQueueProfile:
        profile = cls(link_ids, step_minutes)
        directory = panel_dir / "train" / "mainline_states"
        files = sorted(directory.glob("**/*.parquet")) if directory.is_dir() else []
        if not files:
            raise ProjectError(f"нет train для повторяемости очереди: {directory}")
        for path in files:
            frame = pd.read_parquet(path, columns=["timestamp", "link_id", "speed_kmh"])
            profile.update(frame, thresholds)
        return profile


class QueueHorizon:
    """Правило одного окна: снимок на T, сдвиг вверх по потоку, повторяемость для onset."""

    def __init__(self, settings: QueueSettings) -> None:
        self.settings = settings

    def predict(
        self,
        condition: str,
        template: pd.DataFrame,
        speeds: dict[str, float],
        chain: LinkChain,
        thresholds: dict[str, float],
        recurrent: RecurrentQueueProfile,
    ) -> pd.DataFrame:
        """Вернуть строки шаблона с queue_pred. Метки времени шаблона не переписывает."""
        if condition not in (self.settings.onset_condition, self.settings.ongoing_condition):
            raise ProjectError(f"неизвестное условие окна очереди: {condition}")
        required = {"window_id", "timestamp", "link_id"}
        missing = required - set(template.columns)
        if missing:
            raise ProjectError(f"шаблон очереди не содержит {sorted(missing)}")
        observed = {
            link_id: 0
            for link_id, speed in speeds.items()
            if link_id in thresholds and float(speed) <= thresholds[link_id]
        }
        times = pd.to_datetime(template["timestamp"], utc=True)
        order = sorted(value for value in times.unique() if not pd.isna(value))
        index_of = {value: index for index, value in enumerate(order)}
        prior: dict[str, int] = {}
        if condition == self.settings.onset_condition:
            for link_id in template["link_id"].astype(str).unique():
                if link_id in observed:
                    continue
                earliest = self._earliest_recurrent(link_id, order, recurrent)
                if earliest is not None:
                    prior[link_id] = earliest
        result = template[["window_id", "timestamp", "link_id"]].copy()
        result["window_id"] = result["window_id"].astype(str)
        result["timestamp"] = template["timestamp"].astype(str)
        result["link_id"] = result["link_id"].astype(str)
        prediction = np.zeros(len(result), dtype=np.int64)
        step_index = times.map(index_of)
        for index in range(len(order)):
            active = self._active(observed, prior, index, chain)
            if not active:
                continue
            rows = step_index.eq(index) & result["link_id"].isin(active)
            prediction[rows.to_numpy()] = 1
        result["queue_pred"] = prediction
        return result

    def _earliest_recurrent(
        self,
        link_id: str,
        order: list[pd.Timestamp],
        recurrent: RecurrentQueueProfile,
    ) -> int | None:
        for index, timestamp in enumerate(order):
            share = recurrent.share(link_id, timestamp)
            if share is not None and share > self.settings.recurrent_share_min:
                return index
        return None

    def _active(self, observed: dict[str, int], prior: dict[str, int], index: int, chain: LinkChain) -> set[str]:
        """Наблюдаемая очередь тянет хвост вверх. Повторяемость включает только саму связь."""
        active: set[str] = set()
        for link_id, born in observed.items():
            if born > index:
                continue
            self._walk(link_id, self.settings.hops_after(index - born), chain, active)
        for link_id, born in prior.items():
            if born <= index:
                active.add(link_id)
        return active

    @staticmethod
    def _walk(link_id: str, hops: int, chain: LinkChain, active: set[str]) -> None:
        node: str | None = link_id
        active.add(link_id)
        for _ in range(hops):
            if node is None:
                return
            node = chain.upstream(node)
            if node is None:
                return
            active.add(node)


def speed_thresholds(panel_dir: Path, free_speed_fraction: float) -> dict[str, float]:
    """Порог связи: доля свободной скорости из параметров диаграммы. Пропуск в файле не заменяется числом."""
    path = panel_dir / "network" / "fd_parameters.csv"
    if not path.is_file():
        raise ProjectError(f"нет параметров диаграммы для порога очереди: {path}")
    frame = pd.read_csv(path)
    if "link_id" not in frame.columns or "free_speed_kmh" not in frame.columns:
        raise ProjectError(f"{path} не содержит link_id и free_speed_kmh")
    speed = pd.to_numeric(frame["free_speed_kmh"], errors="coerce")
    valid = speed.notna() & (speed > 0)
    return {
        str(link_id): float(value) * free_speed_fraction
        for link_id, value in zip(frame.loc[valid, "link_id"].astype(str), speed[valid].astype(float))
    }
