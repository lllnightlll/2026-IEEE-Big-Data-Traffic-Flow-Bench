"""Очередь, выученная на train. Признаки кончаются на T, метка — скорость горизонта.

Метка не читается из скрытого файла окон. На train очередь — это скорость
не выше порога связи. Модель заменяет уже принятое правило только если на
отложенных днях, которые она не видела, равное среднее IoU onset и ongoing
выше этого правила на заданный запас.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trafficflowbench.errors import ProjectError
from trafficflowbench.queue.forecast import RecurrentQueueProfile
from trafficflowbench.queue.model_settings import QueueModelSettings
from trafficflowbench.queue.settings import QueueSettings

# Порядок колонок фиксирован: обучение и сдача читают один и тот же вектор.
# Скорости после T здесь нет. Есть только часы будущего шага и доля очереди
# в эти часы на других днях train.
FEATURE_NAMES = (
    "speed_ratio",
    "flow_ratio",
    "occupancy",
    "slope_short",
    "slope_long",
    "min_speed_ratio",
    "queued_now",
    "history_queue_fraction",
    "downstream_speed_ratio",
    "downstream_queued",
    "upstream_speed_ratio",
    "steps_ahead",
    "recurrent_share",
    "minute_fraction",
    "weekday_fraction",
    "position_fraction",
)

# Добавка к вектору, когда onset и ongoing учатся отдельно.
# Всё по-прежнему считается по истории до T включительно.
EXTRA_FEATURE_NAMES = (
    "flow_slope_short",
    "occupancy_slope_short",
    "density_ratio",
    "downstream_flow_ratio",
    "speed_gap_downstream",
    "downstream_queue_proximity",
    "recurrent_share_now",
)


@dataclass
class DayLattice:
    """Один день коридора. Строка — связь в порядке движения, столбец — пятиминутка."""

    link_ids: tuple[str, ...]
    times: pd.DatetimeIndex
    speed: np.ndarray
    flow: np.ndarray
    occupancy: np.ndarray
    free_speed: np.ndarray
    capacity: np.ndarray
    thresholds: np.ndarray

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        link_ids: tuple[str, ...],
        free_speed: np.ndarray,
        capacity: np.ndarray,
        thresholds: np.ndarray,
    ) -> DayLattice | None:
        required = {"timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "is_score_eligible"}
        missing = required - set(frame.columns)
        if missing:
            raise ProjectError(f"в дне очереди нет колонок {sorted(missing)}")
        eligible = pd.to_numeric(frame["is_score_eligible"], errors="coerce").fillna(0.0) >= 0.5
        kept = frame.loc[eligible].copy()
        if kept.empty:
            return None
        kept["timestamp"] = pd.to_datetime(kept["timestamp"], utc=True, format="ISO8601")
        kept["link_id"] = kept["link_id"].astype(str)
        kept["speed_kmh"] = pd.to_numeric(kept["speed_kmh"], errors="coerce")
        kept["flow_vph"] = pd.to_numeric(kept["flow_vph"], errors="coerce")
        kept["occupancy"] = pd.to_numeric(kept["occupancy"], errors="coerce")
        kept = kept.dropna(subset=["timestamp"])
        grouped = kept.groupby(["link_id", "timestamp"], sort=False).agg(
            speed_kmh=("speed_kmh", "min"),
            flow_vph=("flow_vph", "mean"),
            occupancy=("occupancy", "mean"),
        )
        if grouped.empty:
            return None
        times = pd.DatetimeIndex(sorted(grouped.index.get_level_values("timestamp").unique()))
        position = {link_id: index for index, link_id in enumerate(link_ids)}
        speed = np.full((len(link_ids), len(times)), np.nan)
        flow = np.full_like(speed, np.nan)
        occupancy = np.full_like(speed, np.nan)
        link_codes = grouped.index.get_level_values("link_id").astype(str)
        stamps = pd.DatetimeIndex(grouped.index.get_level_values("timestamp"))
        row_index = pd.Series(link_codes).map(position).to_numpy()
        column_index = times.get_indexer(stamps)
        valid = pd.notna(row_index) & (column_index >= 0)
        rows = row_index[valid].astype(np.int64)
        columns = column_index[valid]
        speed[rows, columns] = grouped["speed_kmh"].to_numpy()[valid]
        flow[rows, columns] = grouped["flow_vph"].to_numpy()[valid]
        occupancy[rows, columns] = grouped["occupancy"].to_numpy()[valid]
        return cls(link_ids, times, speed, flow, occupancy, free_speed, capacity, thresholds)


def share_matrix(profile: RecurrentQueueProfile, times: pd.DatetimeIndex) -> np.ndarray:
    """Доля дней с очередью. Строка профиля совпадает со строкой решётки."""
    if len(times) == 0 or profile.seen.shape[0] == 0:
        return np.zeros((profile.seen.shape[0], len(times)))
    weekdays = np.array([int(stamp.weekday()) for stamp in times], dtype=np.int64)
    slots = np.array(
        [(int(stamp.hour) * 60 + int(stamp.minute)) // profile.step_minutes for stamp in times],
        dtype=np.int64,
    )
    seen = profile.seen[:, weekdays, slots]
    queued = profile.queued[:, weekdays, slots]
    out = np.full(seen.shape, np.nan, dtype=float)
    np.divide(queued, seen, out=out, where=seen > 0)
    return out


def origin_batch(
    lattice: DayLattice,
    origin: int,
    history_points: int,
    horizon: int,
    short_steps: int,
    profile: RecurrentQueueProfile,
    extended: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Признаки, метки и маска конечной скорости горизонта для момента origin.

    Срез истории — последние history_points столбцов, заканчивая origin.
    Столбцы правее origin в признаки не входят.
    """
    start = origin - history_points + 1
    if start < 0:
        raise ProjectError("история момента T короче заданного окна")
    future = slice(origin + 1, origin + 1 + horizon)
    speed_hist = lattice.speed[:, start : origin + 1]
    flow_hist = lattice.flow[:, start : origin + 1]
    occ_hist = lattice.occupancy[:, start : origin + 1]
    future_speed = lattice.speed[:, future]
    shares = share_matrix(profile, lattice.times[future])
    shares_now = share_matrix(profile, lattice.times[origin : origin + 1])[:, 0]
    features = feature_matrix(
        speed_hist,
        flow_hist,
        occ_hist,
        shares,
        lattice.times[future],
        lattice,
        short_steps,
        shares_now,
        extended,
    )
    valid = np.isfinite(future_speed)
    labels = valid & (future_speed <= lattice.thresholds[:, None])
    return features, labels.reshape(-1), valid.reshape(-1)


def feature_matrix(
    speed: np.ndarray,
    flow: np.ndarray,
    occupancy: np.ndarray,
    shares: np.ndarray,
    future_times: pd.DatetimeIndex,
    lattice: DayLattice,
    short_steps: int,
    shares_now: np.ndarray,
    extended: bool,
) -> np.ndarray:
    """Один вектор на связь и шаг горизонта. extended добавляет признаки срыва и хвоста."""
    return _features(
        speed,
        flow,
        occupancy,
        shares,
        future_times,
        lattice,
        short_steps,
        shares_now if extended else None,
    )


def _features(
    speed: np.ndarray,
    flow: np.ndarray,
    occupancy: np.ndarray,
    shares: np.ndarray,
    future_times: pd.DatetimeIndex,
    lattice: DayLattice,
    short_steps: int,
    shares_now: np.ndarray | None = None,
) -> np.ndarray:
    width, past = speed.shape
    horizon = shares.shape[1]
    free = lattice.free_speed
    now = speed[:, -1]
    speed_ratio = _divide(now, free)
    flow_ratio = _divide(flow[:, -1], lattice.capacity)
    occ_now = occupancy[:, -1]
    if past > short_steps:
        slope_short = _divide(now - speed[:, -1 - short_steps], free) / short_steps
    else:
        slope_short = np.full(width, np.nan)
    slope_long = _divide(now - speed[:, 0], free) / max(past - 1, 1)
    known_rows = np.isfinite(speed).any(axis=1)
    lowest = np.full(width, np.nan)
    if known_rows.any():
        lowest[known_rows] = np.nanmin(speed[known_rows], axis=1)
    min_ratio = _divide(lowest, free)
    known = np.isfinite(speed)
    queued = known & (speed <= lattice.thresholds[:, None])
    known_count = known.sum(axis=1)
    history_fraction = np.divide(queued.sum(axis=1), known_count, out=np.full(width, np.nan), where=known_count > 0)
    queued_now = known[:, -1] & (now <= lattice.thresholds)
    downstream_ratio = np.full(width, np.nan)
    downstream_queued = np.zeros(width, dtype=float)
    upstream_ratio = np.full(width, np.nan)
    if width > 1:
        downstream_ratio[:-1] = speed_ratio[1:]
        downstream_queued[:-1] = queued_now[1:].astype(float)
        upstream_ratio[1:] = speed_ratio[:-1]
    position = np.linspace(0.0, 1.0, width) if width > 1 else np.zeros(width)
    minutes = np.array([stamp.hour * 60 + stamp.minute for stamp in future_times], dtype=float) / (24.0 * 60.0)
    weekdays = np.array([stamp.weekday() for stamp in future_times], dtype=float) / 6.0
    steps = np.tile(np.arange(1, horizon + 1, dtype=float), width)
    columns = [
        np.repeat(speed_ratio, horizon),
        np.repeat(flow_ratio, horizon),
        np.repeat(occ_now, horizon),
        np.repeat(slope_short, horizon),
        np.repeat(slope_long, horizon),
        np.repeat(min_ratio, horizon),
        np.repeat(queued_now.astype(float), horizon),
        np.repeat(history_fraction, horizon),
        np.repeat(downstream_ratio, horizon),
        np.repeat(downstream_queued, horizon),
        np.repeat(upstream_ratio, horizon),
        steps,
        shares.reshape(-1),
        np.tile(minutes, width),
        np.tile(weekdays, width),
        np.repeat(position, horizon),
    ]
    if shares_now is not None:
        extra = _link_extras(speed, flow, occupancy, lattice, short_steps, shares_now, queued_now, flow_ratio, speed_ratio)
        for column in range(extra.shape[1]):
            columns.append(np.repeat(extra[:, column], horizon))
    return np.column_stack(columns)


def _link_extras(
    speed: np.ndarray,
    flow: np.ndarray,
    occupancy: np.ndarray,
    lattice: DayLattice,
    short_steps: int,
    shares_now: np.ndarray,
    queued_now: np.ndarray,
    flow_ratio: np.ndarray,
    speed_ratio: np.ndarray,
) -> np.ndarray:
    """Признаки связи на момент T. Повторяются на каждый шаг горизонта снаружи."""
    width, past = speed.shape
    flow_now = flow[:, -1]
    occ_now = occupancy[:, -1]
    if past > short_steps:
        flow_slope = _divide(flow_now - flow[:, -1 - short_steps], lattice.capacity) / short_steps
        occupancy_slope = _divide(occ_now - occupancy[:, -1 - short_steps], np.ones(width)) / short_steps
    else:
        flow_slope = np.full(width, np.nan)
        occupancy_slope = np.full(width, np.nan)
    # Плотность относительно свободного режима: (q/v) / (capacity/free_speed).
    # Около единицы связь загружена, хотя скорость ещё свободная.
    density_ratio = _divide(flow_now * lattice.free_speed, speed[:, -1] * lattice.capacity)
    downstream_flow = np.full(width, np.nan)
    downstream_speed = np.full(width, np.nan)
    if width > 1:
        downstream_flow[:-1] = flow_ratio[1:]
        downstream_speed[:-1] = speed_ratio[1:]
    speed_gap = speed_ratio - downstream_speed
    proximity = downstream_queue_proximity(queued_now)
    return np.column_stack(
        [
            flow_slope,
            occupancy_slope,
            density_ratio,
            downstream_flow,
            speed_gap,
            proximity,
            shares_now,
        ]
    )


def downstream_queue_proximity(queued_now: np.ndarray) -> np.ndarray:
    """Насколько близко по ходу движения уже стоит очередь. Своя очередь даёт 1, отсутствие — 0."""
    width = int(queued_now.shape[0])
    nearest = np.full(width, np.inf)
    nearest[np.asarray(queued_now, dtype=bool)] = 0.0
    for index in range(width - 2, -1, -1):
        nearest[index] = min(float(nearest[index]), float(nearest[index + 1]) + 1.0)
    proximity = np.zeros(width, dtype=float)
    finite = np.isfinite(nearest)
    proximity[finite] = 1.0 / (1.0 + nearest[finite])
    return proximity


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(np.broadcast(numerator, denominator).shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=out, where=np.isfinite(denominator) & (denominator != 0) & np.isfinite(numerator))
    return out


class ObservedQueueRule:
    """То же правило, что уже принято: снимок на T, сдвиг вверх, повторяемость для onset."""

    def __init__(self, queue: QueueSettings) -> None:
        self.queue = queue

    def predict(self, observed: np.ndarray, shares: np.ndarray, onset: bool) -> np.ndarray:
        width, horizon = shares.shape
        prediction = np.zeros((width, horizon), dtype=bool)
        seeds = np.flatnonzero(observed)
        for step in range(horizon):
            hops = self.queue.hops_after(step)
            active = np.zeros(width, dtype=bool)
            for seed in seeds:
                node = int(seed)
                active[node] = True
                for _ in range(hops):
                    if node == 0:
                        break
                    node -= 1
                    active[node] = True
            prediction[:, step] = active
        if onset:
            known = np.isfinite(shares)
            prediction = prediction | (known & (shares > self.queue.recurrent_share_min))
        return prediction


@dataclass
class ProxyWindow:
    """Одно окно отложенного дня. Вероятности ещё не превращены в 0/1."""

    split: str
    condition: str
    truth: np.ndarray
    persistence: np.ndarray
    rule: np.ndarray
    probability: np.ndarray
    split_probability: np.ndarray | None = None


def window_iou(truth: np.ndarray, prediction: np.ndarray) -> float:
    intersection = int(np.logical_and(truth, prediction).sum())
    union = int(np.logical_or(truth, prediction).sum())
    if union == 0:
        return 1.0
    return intersection / union


def equal_mean(
    windows: list[ProxyWindow],
    prediction_of,
    min_windows: int,
    conditions: tuple[str, str],
) -> float | None:
    """Среднее IoU каждого условия, затем их равное среднее. Как у организатора."""
    grouped: dict[str, list[float]] = {condition: [] for condition in conditions}
    for window in windows:
        if window.condition not in grouped:
            continue
        grouped[window.condition].append(window_iou(window.truth, prediction_of(window)))
    means = [float(np.mean(values)) for values in grouped.values() if len(values) >= min_windows]
    if not means:
        return None
    return float(np.mean(means))


class BalancedDraw:
    """Держит не больше заданного числа единиц и нулей. Единицы редки, без потолка они утонут."""

    def __init__(self, max_positive: int, max_negative: int, rng: np.random.Generator) -> None:
        self.max_positive = max_positive
        self.max_negative = max_negative
        self.rng = rng
        self._positive: list[np.ndarray] = []
        self._negative: list[np.ndarray] = []
        self.positive_rows = 0
        self.negative_rows = 0

    def add(self, features: np.ndarray, labels: np.ndarray) -> None:
        if features.size == 0:
            return
        positive = labels.astype(bool)
        if positive.any():
            self._positive.append(np.ascontiguousarray(features[positive]))
            self.positive_rows += int(positive.sum())
        if (~positive).any():
            self._negative.append(np.ascontiguousarray(features[~positive]))
            self.negative_rows += int((~positive).sum())
        if self.positive_rows > self.max_positive * 2 or self.negative_rows > self.max_negative * 2:
            self._trim()

    def finish(self) -> tuple[np.ndarray, np.ndarray] | None:
        self._trim()
        if self.positive_rows == 0 or self.negative_rows == 0:
            return None
        features = np.vstack(self._positive + self._negative)
        labels = np.concatenate(
            [np.ones(self.positive_rows, dtype=np.int8), np.zeros(self.negative_rows, dtype=np.int8)]
        )
        return features, labels

    def _trim(self) -> None:
        self._positive, self.positive_rows = self._keep(self._positive, self.max_positive)
        self._negative, self.negative_rows = self._keep(self._negative, self.max_negative)

    def _keep(self, chunks: list[np.ndarray], limit: int) -> tuple[list[np.ndarray], int]:
        if not chunks:
            return [], 0
        stacked = np.vstack(chunks) if len(chunks) > 1 else chunks[0]
        if len(stacked) > limit:
            chosen = self.rng.choice(len(stacked), size=limit, replace=False)
            stacked = stacked[np.sort(chosen)]
        return [stacked], len(stacked)


class TrainedQueueModel:
    """Градиентный бустинг по фиксированному вектору. Порог вероятности выбирается снаружи."""

    def __init__(self, estimator, threshold: float) -> None:
        self.estimator = estimator
        self.threshold = threshold

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, settings: QueueModelSettings, threshold: float) -> TrainedQueueModel:
        from sklearn.ensemble import HistGradientBoostingClassifier

        estimator = HistGradientBoostingClassifier(
            learning_rate=settings.learning_rate,
            max_iter=settings.max_iter,
            max_leaf_nodes=settings.max_leaf_nodes,
            min_samples_leaf=settings.min_samples_leaf,
            l2_regularization=settings.l2_regularization,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=settings.random_state,
        )
        estimator.fit(features, labels)
        return cls(estimator, threshold)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.zeros(0, dtype=float)
        probability = self.estimator.predict_proba(features)
        classes = list(self.estimator.classes_)
        if 1 not in classes:
            return np.zeros(len(features), dtype=float)
        return probability[:, classes.index(1)]

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.probabilities(features) >= self.threshold


class SplitQueueModel:
    """Две модели: срыв очереди и уже идущая очередь. У каждой свой порог вероятности.

    Если на одном из условий положительных примеров не хватило, этот ответ
    берётся из общей модели, обученной на всех окнах.
    """

    def __init__(
        self,
        onset: TrainedQueueModel | None,
        ongoing: TrainedQueueModel | None,
        fallback: TrainedQueueModel | None,
        onset_threshold: float,
        ongoing_threshold: float,
    ) -> None:
        self.onset = onset
        self.ongoing = ongoing
        self.fallback = fallback
        self.onset_threshold = onset_threshold
        self.ongoing_threshold = ongoing_threshold

    def probabilities(self, features: np.ndarray, onset: bool) -> np.ndarray:
        specialist = self.onset if onset else self.ongoing
        if specialist is None:
            if self.fallback is None:
                return np.zeros(len(features), dtype=float)
            return self.fallback.probabilities(features)
        return specialist.probabilities(features)

    def threshold_for(self, onset: bool) -> float:
        return self.onset_threshold if onset else self.ongoing_threshold

    def ready(self) -> bool:
        return self.onset is not None or self.ongoing is not None or self.fallback is not None
