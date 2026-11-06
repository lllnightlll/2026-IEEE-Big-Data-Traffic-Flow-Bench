"""Поправка скорости и потока поверх контекста.

Контекст заполняет дырку линейным остатком соседей. На скачке скорости это
смешивает затор со свободным потоком. Дерево учит добавку только на вырезанных
ячейках и только по тому, что видно в маскированном слое. Наблюдаемая ячейка
не меняется: её значение уже совпадает с детектором.

Локальный S_state повторяет опубликованную формулу: RMSE скорости к 25 км/ч,
RMSE потока на полосу к 600 авт/ч, веса 0.54 и 0.46, затем равное среднее режимов.
Официальный score_task1.py на частичном файле не вызывается: пропущенные строки
он считает нулями и портит знаменатель.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trafficflowbench.config_loader import ScoringContract, _as_float, _as_int, _read_mapping
from trafficflowbench.errors import ConfigError, ProjectError
from trafficflowbench.state.context_model import ContextStateModel, _date_from_name
from trafficflowbench.state.corridor import DiagramBounds, LinkChain
from trafficflowbench.state.settings import ContextSettings

FEATURE_NAMES = (
    "context_speed",
    "context_flow",
    "profile_speed",
    "profile_flow",
    "speed_minus_profile",
    "flow_minus_profile",
    "neighbor_min_speed",
    "neighbor_mean_flow",
    "neighbor_observed_fraction",
    "temporal_min_speed",
    "hour_fraction",
    "weekday_fraction",
    "position_fraction",
    "free_speed",
    "capacity",
    "jam_density",
    "lanes",
)

MASKED_COLUMNS = ["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "pct_observed"]
TRUTH_COLUMNS = ["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]


@dataclass(frozen=True)
class StateResidualSettings:
    """Сколько строк учить и какой прирост S_state считать достаточным."""

    max_train_rows: int
    rows_per_day: int
    min_state_gain: float
    missing_lane_count: float
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    random_state: int
    artifact_dir: str
    queue_candidates: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> StateResidualSettings:
        raw = _read_mapping(path)
        candidates = raw.get("queue_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ConfigError("queue_candidates должен быть непустым списком путей")
        settings = cls(
            max_train_rows=_as_int(raw.get("max_train_rows"), "max_train_rows"),
            rows_per_day=_as_int(raw.get("rows_per_day"), "rows_per_day"),
            min_state_gain=_as_float(raw.get("min_state_gain"), "min_state_gain"),
            missing_lane_count=_as_float(raw.get("missing_lane_count"), "missing_lane_count"),
            learning_rate=_as_float(raw.get("learning_rate"), "learning_rate"),
            max_iter=_as_int(raw.get("max_iter"), "max_iter"),
            max_leaf_nodes=_as_int(raw.get("max_leaf_nodes"), "max_leaf_nodes"),
            min_samples_leaf=_as_int(raw.get("min_samples_leaf"), "min_samples_leaf"),
            l2_regularization=_as_float(raw.get("l2_regularization"), "l2_regularization"),
            random_state=_as_int(raw.get("random_state"), "random_state"),
            artifact_dir=str(raw.get("artifact_dir") or ""),
            queue_candidates=tuple(str(item) for item in candidates),
        )
        if settings.max_train_rows <= 0 or settings.rows_per_day <= 0:
            raise ConfigError("размеры выборки State должны быть положительными")
        if settings.min_state_gain < 0:
            raise ConfigError("min_state_gain не может быть отрицательным")
        if settings.missing_lane_count <= 0:
            raise ConfigError("missing_lane_count должен быть положительным")
        if settings.learning_rate <= 0 or settings.max_iter <= 0 or settings.max_leaf_nodes <= 1:
            raise ConfigError("параметры дерева State заданы неверно")
        if settings.min_samples_leaf <= 0 or settings.l2_regularization < 0:
            raise ConfigError("лист и регуляризация State заданы неверно")
        if not settings.artifact_dir:
            raise ConfigError("state_residual.artifact_dir должен быть путём")
        return settings


class LaneBook:
    """Число полос станции и связи. Поток в S_state делится на это число, как у организатора."""

    def __init__(self, by_key: dict[str, float], missing: float) -> None:
        self.by_key = by_key
        self.missing = missing

    @classmethod
    def read(cls, panel_dir: Path, missing: float) -> LaneBook:
        by_key: dict[str, float] = {}

        def absorb(frame: pd.DataFrame, key: str) -> None:
            if key not in frame.columns or "lanes" not in frame.columns:
                return
            lanes = pd.to_numeric(frame["lanes"], errors="coerce")
            # 99 в links.csv — соединитель зоны, а не полоса детектора.
            valid = lanes.notna() & (lanes >= 1) & (lanes <= 16)
            by_key.update(zip(frame.loc[valid, key].astype(str), lanes[valid].astype(float)))

        links = panel_dir / "network" / "links.csv"
        fd = panel_dir / "network" / "fd_parameters.csv"
        if links.is_file():
            absorb(pd.read_csv(links, dtype={"link_id": str}), "link_id")
        if fd.is_file():
            frame = pd.read_csv(fd, dtype={"station_id": str, "link_id": str})
            absorb(frame, "link_id")
            absorb(frame, "station_id")
        if not by_key:
            raise ProjectError(f"нет числа полос в сети {panel_dir.name}")
        return cls(by_key, missing)

    def counts(self, stations: np.ndarray, links: np.ndarray) -> np.ndarray:
        by_station = pd.Series(stations).map(self.by_key)
        by_link = pd.Series(links).map(self.by_key)
        return by_station.fillna(by_link).fillna(self.missing).clip(lower=self.missing).to_numpy(dtype=float)


class StateCellScore:
    """Накопитель RMSE по режимам маски. Пустой режим даёт 0, как пустая панель у организатора."""

    def __init__(self, scoring: ScoringContract) -> None:
        self.scoring = scoring
        self.sq_speed = {regime: 0.0 for regime in scoring.regimes}
        self.sq_flow = {regime: 0.0 for regime in scoring.regimes}
        self.count = {regime: 0 for regime in scoring.regimes}

    def add(
        self,
        regime: str,
        pred_speed: np.ndarray,
        pred_flow: np.ndarray,
        true_speed: np.ndarray,
        true_flow: np.ndarray,
        lanes: np.ndarray,
    ) -> None:
        if regime not in self.count:
            raise ProjectError(f"неизвестный режим маски {regime}")
        usable = (
            np.isfinite(pred_speed)
            & np.isfinite(pred_flow)
            & np.isfinite(true_speed)
            & np.isfinite(true_flow)
            & np.isfinite(lanes)
            & (lanes > 0)
        )
        if not usable.any():
            return
        speed_gap = pred_speed[usable] - true_speed[usable]
        flow_gap = (pred_flow[usable] - true_flow[usable]) / lanes[usable]
        self.sq_speed[regime] += float(np.sum(speed_gap * speed_gap))
        self.sq_flow[regime] += float(np.sum(flow_gap * flow_gap))
        self.count[regime] += int(usable.sum())

    def value(self) -> float:
        parts = [self.regime_value(regime) for regime in self.scoring.regimes]
        if not parts:
            return 0.0
        return float(np.mean(parts))

    def regime_value(self, regime: str) -> float:
        count = self.count[regime]
        if count == 0:
            return 0.0
        rmse_speed = float(np.sqrt(self.sq_speed[regime] / count))
        rmse_flow = float(np.sqrt(self.sq_flow[regime] / count))
        speed_score = max(0.0, 1.0 - rmse_speed / self.scoring.speed_rmse_scale_kmh)
        flow_score = max(0.0, 1.0 - rmse_flow / self.scoring.flow_rmse_scale_vph_per_lane)
        return self.scoring.state_speed_weight * speed_score + self.scoring.state_flow_weight * flow_score


def neighbor_min_speed(grid: np.ndarray, hops: int) -> np.ndarray:
    """Минимум наблюдаемой скорости соседей по коридору. Своя ячейка в минимум не входит.

    grid имеет форму (время, связь). Пустые соседи остаются NaN, дерево умеет такой пропуск.
    """
    if hops < 0:
        raise ProjectError("число соседей не может быть отрицательным")
    found = np.full(grid.shape, np.nan)
    if hops == 0 or grid.size == 0:
        return found
    for hop in range(1, hops + 1):
        left = np.full(grid.shape, np.nan)
        right = np.full(grid.shape, np.nan)
        if hop < grid.shape[1]:
            left[:, hop:] = grid[:, :-hop]
            right[:, :-hop] = grid[:, hop:]
        found = _nanmin(found, left)
        found = _nanmin(found, right)
    return found


def temporal_min_speed(grid: np.ndarray, steps: int) -> np.ndarray:
    """Минимум скорости той же связи на соседних пятиминутках, без самой ячейки."""
    if steps < 0:
        raise ProjectError("temporal_steps не может быть отрицательным")
    found = np.full(grid.shape, np.nan)
    if steps == 0 or grid.size == 0:
        return found
    for step in range(1, steps + 1):
        earlier = np.full(grid.shape, np.nan)
        later = np.full(grid.shape, np.nan)
        if step < grid.shape[0]:
            earlier[step:, :] = grid[:-step, :]
            later[:-step, :] = grid[step:, :]
        found = _nanmin(found, earlier)
        found = _nanmin(found, later)
    return found


def _nanmin(current: np.ndarray, addition: np.ndarray) -> np.ndarray:
    both = np.isfinite(current) & np.isfinite(addition)
    only_new = ~np.isfinite(current) & np.isfinite(addition)
    out = current.copy()
    out[both] = np.minimum(current[both], addition[both])
    out[only_new] = addition[only_new]
    return out


class ResidualCorrector:
    """Две регрессии: добавка к скорости и добавка к потоку. Наблюдаемые ячейки не трогает."""

    def __init__(self, speed_model, flow_model) -> None:
        self.speed_model = speed_model
        self.flow_model = flow_model

    @classmethod
    def fit(cls, features: np.ndarray, speed_residual: np.ndarray, flow_residual: np.ndarray, settings: StateResidualSettings) -> ResidualCorrector:
        from sklearn.ensemble import HistGradientBoostingRegressor

        def one(target: np.ndarray):
            estimator = HistGradientBoostingRegressor(
                learning_rate=settings.learning_rate,
                max_iter=settings.max_iter,
                max_leaf_nodes=settings.max_leaf_nodes,
                min_samples_leaf=settings.min_samples_leaf,
                l2_regularization=settings.l2_regularization,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=settings.random_state,
            )
            estimator.fit(features, target)
            return estimator

        return cls(one(speed_residual), one(flow_residual))

    def apply(
        self,
        features: np.ndarray,
        context_speed: np.ndarray,
        context_flow: np.ndarray,
        hidden: np.ndarray,
        diagram: DiagramBounds,
        link_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        speed = np.array(context_speed, dtype=float, copy=True)
        flow = np.array(context_flow, dtype=float, copy=True)
        if hidden.any():
            speed[hidden] = speed[hidden] + self.speed_model.predict(features[hidden])
            flow[hidden] = flow[hidden] + self.flow_model.predict(features[hidden])
        return diagram.clip(link_ids, speed, flow)


@dataclass
class DayTables:
    """Один день, уже сведённый к строкам маски. hidden — ячейка вырезана и в обоих полях пуста."""

    regime: str
    timestamp: pd.Series
    station_id: np.ndarray
    link_id: np.ndarray
    context_speed: np.ndarray
    context_flow: np.ndarray
    true_speed: np.ndarray
    true_flow: np.ndarray
    hidden: np.ndarray
    eligible: np.ndarray
    lanes: np.ndarray


class ResidualPanelStudy:
    """Учит поправку одного коридора и считает S_state на отложенных днях."""

    def __init__(
        self,
        panel_dir: Path,
        context: ContextSettings,
        residual: StateResidualSettings,
        scoring: ScoringContract,
        chain: LinkChain,
        diagram: DiagramBounds,
    ) -> None:
        self.panel_dir = panel_dir
        self.context = context
        self.residual = residual
        self.scoring = scoring
        self.chain = chain
        self.diagram = diagram
        self.lanes = LaneBook.read(panel_dir, residual.missing_lane_count)

    def train(self, model: ContextStateModel, skip_dates: set[str]) -> ResidualCorrector | None:
        features, speed_residual, flow_residual = self._sample(model, skip_dates, keep_dates=None)
        if features is None or len(features) < self.residual.min_samples_leaf * 2:
            return None
        return ResidualCorrector.fit(features, speed_residual, flow_residual, self.residual)

    def score(self, model: ContextStateModel, corrector: ResidualCorrector, keep_dates: set[str]) -> tuple[float, float]:
        context_score = StateCellScore(self.scoring)
        corrected_score = StateCellScore(self.scoring)
        for path, truth_path, regime, date_text in self._days():
            if date_text not in keep_dates:
                continue
            day = self._day(path, truth_path, regime, model)
            if day is None:
                continue
            target = day.hidden & day.eligible & np.isfinite(day.true_speed) & np.isfinite(day.true_flow)
            if not target.any():
                continue
            built = self._features(day, model)
            corrected_speed, corrected_flow = corrector.apply(
                built,
                day.context_speed,
                day.context_flow,
                day.hidden,
                self.diagram,
                day.link_id,
            )
            context_score.add(regime, day.context_speed[target], day.context_flow[target], day.true_speed[target], day.true_flow[target], day.lanes[target])
            corrected_score.add(regime, corrected_speed[target], corrected_flow[target], day.true_speed[target], day.true_flow[target], day.lanes[target])
        return context_score.value(), corrected_score.value()

    def predict_frame(self, model: ContextStateModel, corrector: ResidualCorrector, frame: pd.DataFrame) -> pd.DataFrame:
        """Прогноз для кадра без истины. Метку времени не переписывает."""
        predicted = model.predict_observations(frame)
        hidden = pd.to_numeric(frame["speed_kmh"], errors="coerce").isna().to_numpy() & pd.to_numeric(
            frame["flow_vph"], errors="coerce"
        ).isna().to_numpy()
        day = DayTables(
            regime="",
            timestamp=pd.to_datetime(predicted["timestamp"], utc=True, format="ISO8601"),
            station_id=predicted["station_id"].astype(str).to_numpy(),
            link_id=predicted["link_id"].astype(str).to_numpy(),
            context_speed=predicted["speed_kmh"].to_numpy(dtype=float),
            context_flow=predicted["flow_vph"].to_numpy(dtype=float),
            true_speed=np.full(len(predicted), np.nan),
            true_flow=np.full(len(predicted), np.nan),
            hidden=hidden,
            eligible=np.zeros(len(predicted), dtype=bool),
            lanes=self.lanes.counts(predicted["station_id"].astype(str).to_numpy(), predicted["link_id"].astype(str).to_numpy()),
        )
        features = self._features(day, model)
        speed, flow = corrector.apply(features, day.context_speed, day.context_flow, hidden, self.diagram, day.link_id)
        out = predicted.copy()
        out["speed_kmh"] = speed
        out["flow_vph"] = flow
        return out

    def _sample(
        self,
        model: ContextStateModel,
        skip_dates: set[str],
        keep_dates: set[str] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        rng = np.random.default_rng(self.residual.random_state)
        chunks: list[np.ndarray] = []
        speed_parts: list[np.ndarray] = []
        flow_parts: list[np.ndarray] = []
        for path, truth_path, regime, date_text in self._days():
            if date_text in skip_dates:
                continue
            if keep_dates is not None and date_text not in keep_dates:
                continue
            day = self._day(path, truth_path, regime, model)
            if day is None:
                continue
            target = np.flatnonzero(day.hidden & day.eligible & np.isfinite(day.true_speed) & np.isfinite(day.true_flow))
            if target.size == 0:
                continue
            if target.size > self.residual.rows_per_day:
                target = rng.choice(target, size=self.residual.rows_per_day, replace=False)
            features = self._features(day, model)[target]
            chunks.append(features)
            speed_parts.append(day.true_speed[target] - day.context_speed[target])
            flow_parts.append(day.true_flow[target] - day.context_flow[target])
        if not chunks:
            return None, None, None
        features = np.vstack(chunks)
        speed_residual = np.concatenate(speed_parts)
        flow_residual = np.concatenate(flow_parts)
        finite = np.isfinite(features).any(axis=1) & np.isfinite(speed_residual) & np.isfinite(flow_residual)
        features = features[finite]
        speed_residual = speed_residual[finite]
        flow_residual = flow_residual[finite]
        if len(features) > self.residual.max_train_rows:
            chosen = np.sort(rng.choice(len(features), size=self.residual.max_train_rows, replace=False))
            features = features[chosen]
            speed_residual = speed_residual[chosen]
            flow_residual = flow_residual[chosen]
        return features, speed_residual, flow_residual

    def _features(self, day: DayTables, model: ContextStateModel) -> np.ndarray:
        link_ids = day.link_id
        profile_speed, profile_flow = model.profile.predict(link_ids, day.timestamp)
        position = pd.Series(link_ids).map(self.chain.position)
        width = max(len(self.chain.link_ids), 1)
        position_fraction = position.fillna(0).to_numpy(dtype=float) / max(width - 1, 1)
        clock = day.timestamp
        hour_fraction = (clock.dt.hour.to_numpy(dtype=float) * 60.0 + clock.dt.minute.to_numpy(dtype=float)) / (24.0 * 60.0)
        weekday_fraction = clock.dt.weekday.to_numpy(dtype=float) / 6.0
        free = np.array([self.diagram.free_speed_kmh.get(link, np.nan) for link in link_ids], dtype=float)
        capacity = np.array([self.diagram.capacity_vph.get(link, np.nan) for link in link_ids], dtype=float)
        jam = np.array([self.diagram.jam_density_per_km.get(link, np.nan) for link in link_ids], dtype=float)
        neighbor_min, neighbor_flow, neighbor_fraction, temporal_min = self._neighbor_columns(day)
        return np.column_stack(
            [
                day.context_speed,
                day.context_flow,
                profile_speed,
                profile_flow,
                day.context_speed - profile_speed,
                day.context_flow - profile_flow,
                neighbor_min,
                neighbor_flow,
                neighbor_fraction,
                temporal_min,
                hour_fraction,
                weekday_fraction,
                position_fraction,
                free,
                capacity,
                jam,
                day.lanes,
            ]
        )

    def _neighbor_columns(self, day: DayTables) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Соседние наблюдаемые значения. Вырезанная ячейка соседом не считается."""
        stamps = pd.DatetimeIndex(day.timestamp)
        times = pd.DatetimeIndex(stamps.unique())
        n_links = len(self.chain.link_ids)
        # +inf, чтобы minimum.at взял первую же конечную скорость, а не застрял на NaN.
        speed_grid = np.full((len(times), n_links), np.inf)
        flow_sum = np.zeros((len(times), n_links))
        flow_count = np.zeros((len(times), n_links))
        observed = ~day.hidden & np.isfinite(day.context_speed) & np.isfinite(day.context_flow)
        positions = pd.Series(day.link_id).map(self.chain.position).to_numpy()
        rows = times.get_indexer(stamps)
        known = observed & pd.notna(positions) & (rows >= 0)
        if known.any():
            row = rows[known]
            column = positions[known].astype(np.int64)
            # Несколько станций одной связи: скорость — минимум, поток — среднее.
            np.minimum.at(speed_grid, (row, column), day.context_speed[known])
            np.add.at(flow_sum, (row, column), day.context_flow[known])
            np.add.at(flow_count, (row, column), 1.0)
        speed_grid[~np.isfinite(speed_grid)] = np.nan
        flow_grid = np.full_like(speed_grid, np.nan)
        np.divide(flow_sum, flow_count, out=flow_grid, where=flow_count > 0)
        speed_min = neighbor_min_speed(speed_grid, self.context.spatial_hops)
        flow_neighbors, seen = _neighbor_mean(flow_grid, self.context.spatial_hops)
        temporal = temporal_min_speed(speed_grid, self.context.temporal_steps)
        out_min = np.full(len(day.link_id), np.nan)
        out_flow = np.full(len(day.link_id), np.nan)
        out_seen = np.zeros(len(day.link_id))
        out_temporal = np.full(len(day.link_id), np.nan)
        usable = pd.notna(positions) & (rows >= 0)
        if usable.any():
            out_min[usable] = speed_min[rows[usable], positions[usable].astype(np.int64)]
            out_flow[usable] = flow_neighbors[rows[usable], positions[usable].astype(np.int64)]
            out_seen[usable] = seen[rows[usable], positions[usable].astype(np.int64)]
            out_temporal[usable] = temporal[rows[usable], positions[usable].astype(np.int64)]
        return out_min, out_flow, out_seen, out_temporal

    def _day(self, masked_path: Path, truth_path: Path, regime: str, model: ContextStateModel) -> DayTables | None:
        masked = pd.read_parquet(masked_path, columns=MASKED_COLUMNS)
        if masked.empty:
            return None
        predicted = model.predict_observations(masked)
        truth = pd.read_parquet(truth_path, columns=TRUTH_COLUMNS)
        predicted = predicted.copy()
        predicted["_time"] = pd.to_datetime(predicted["timestamp"], utc=True, format="ISO8601")
        truth = truth.copy()
        truth["_time"] = pd.to_datetime(truth["timestamp"], utc=True, format="ISO8601")
        truth["station_id"] = truth["station_id"].astype(str)
        truth["link_id"] = truth["link_id"].astype(str)
        truth = truth.drop_duplicates(["_time", "station_id", "link_id"], keep="first")
        merged = predicted.merge(
            truth[["_time", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]],
            on=["_time", "station_id", "link_id"],
            how="left",
            suffixes=("", "_true"),
        )
        hidden = pd.to_numeric(masked["speed_kmh"], errors="coerce").isna().to_numpy() & pd.to_numeric(
            masked["flow_vph"], errors="coerce"
        ).isna().to_numpy()
        if len(hidden) != len(merged):
            raise ProjectError(f"после склейки дня {masked_path.name} число строк изменилось")
        eligible = pd.to_numeric(merged["is_score_eligible"], errors="coerce").fillna(0).to_numpy() >= 0.5
        return DayTables(
            regime=regime,
            timestamp=merged["_time"],
            station_id=merged["station_id"].astype(str).to_numpy(),
            link_id=merged["link_id"].astype(str).to_numpy(),
            context_speed=merged["speed_kmh"].to_numpy(dtype=float),
            context_flow=merged["flow_vph"].to_numpy(dtype=float),
            true_speed=pd.to_numeric(merged["speed_kmh_true"], errors="coerce").to_numpy(dtype=float),
            true_flow=pd.to_numeric(merged["flow_vph_true"], errors="coerce").to_numpy(dtype=float),
            hidden=hidden,
            eligible=eligible,
            lanes=self.lanes.counts(merged["station_id"].astype(str).to_numpy(), merged["link_id"].astype(str).to_numpy()),
        )

    def _days(self) -> list[tuple[Path, Path, str, str]]:
        truth_root = self.panel_dir / "train" / "mainline_states"
        masked_root = self.panel_dir / "train" / "mainline_states_masked"
        truth_by_name = {path.name: path for path in sorted(truth_root.glob("**/*.parquet"))}
        found: list[tuple[Path, Path, str, str]] = []
        for path in sorted(masked_root.glob("**/*.parquet")):
            parent = path.parent.name
            if not parent.startswith("mask_regime="):
                continue
            regime = parent.split("=", 1)[1]
            if regime not in self.scoring.regimes:
                continue
            truth_path = truth_by_name.get(path.name)
            date_text = _date_from_name(path)
            if truth_path is None or date_text is None:
                continue
            found.append((path, truth_path, regime, date_text))
        if not found:
            raise ProjectError(f"нет пар маски и истины для State: {self.panel_dir.name}")
        return found


def _neighbor_mean(grid: np.ndarray, hops: int) -> tuple[np.ndarray, np.ndarray]:
    """Среднее конечных соседей и доля заполненных позиций в окне, без самой связи."""
    total = np.zeros(grid.shape)
    count = np.zeros(grid.shape)
    slots = 0
    for hop in range(1, hops + 1):
        slots += 2
        for shift, target in ((hop, "left"), (-hop, "right")):
            if abs(shift) >= grid.shape[1]:
                continue
            if shift > 0:
                source = grid[:, :-shift]
                dest = (slice(None), slice(shift, None))
            else:
                source = grid[:, -shift:]
                dest = (slice(None), slice(None, shift))
            finite = np.isfinite(source)
            total[dest] += np.where(finite, source, 0.0)
            count[dest] += finite.astype(float)
            del target
    mean = np.full(grid.shape, np.nan)
    np.divide(total, count, out=mean, where=count > 0)
    fraction = count / slots if slots else np.zeros(grid.shape)
    return mean, fraction
