"""Восстановление State по профилю дня и остатку от живых соседей.

Наблюдение берётся только из маскированного слоя: вырезанная ячейка не подсказывает сама себе.
Соседи по коридору и соседние пятиминутки той же связи передают остаток относительно профиля.
Дальше пара скорость–поток обрезается границами диаграммы связи.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trafficflowbench.config_loader import ReleaseLayout
from trafficflowbench.errors import ProjectError
from trafficflowbench.state.corridor import DiagramBounds, LinkChain
from trafficflowbench.state.profile import WeekdayProfile
from trafficflowbench.state.settings import ContextSettings


def observation_weights(frame: pd.DataFrame, settings: ContextSettings) -> np.ndarray:
    """Вес доверия по pct_observed. Конечная скорость без доли наблюдения получает нижний вес, а не ноль."""
    percent = pd.to_numeric(frame["pct_observed"], errors="coerce").to_numpy(dtype=float)
    speed = pd.to_numeric(frame["speed_kmh"], errors="coerce").to_numpy(dtype=float)
    flow = pd.to_numeric(frame["flow_vph"], errors="coerce").to_numpy(dtype=float)
    scaled = np.where(np.isfinite(percent), percent / settings.observation_weight_scale, settings.minimum_observation_weight)
    weight = np.maximum(scaled, settings.minimum_observation_weight)
    usable = np.isfinite(speed) & np.isfinite(flow)
    return np.where(usable, weight, 0.0)


def fill_along_axis(values: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Заполнить пропуски линейно, не дальше чем на max_gap шагов от известной точки.

    values — двумерная решётка. Ось 1 считается направлением заполнения.
    Возвращает заполненные значения и маску ячеек, куда сосед действительно дотянулся.
    """
    if max_gap < 0:
        raise ValueError("max_gap не может быть отрицательным")
    out = np.zeros(values.shape, dtype=float)
    supplied = np.zeros(values.shape, dtype=bool)
    for row_index in range(values.shape[0]):
        filled, reached = _fill_line(values[row_index], max_gap)
        out[row_index] = filled
        supplied[row_index] = reached
    return out, supplied


def _fill_line(values: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    known_at = np.flatnonzero(np.isfinite(values))
    out = np.zeros(values.shape[0], dtype=float)
    reached = np.zeros(values.shape[0], dtype=bool)
    if known_at.size == 0:
        return out, reached
    positions = np.arange(values.shape[0])
    filled = np.interp(positions, known_at, values[known_at])
    insert_at = np.searchsorted(known_at, positions)
    left = np.clip(insert_at - 1, 0, known_at.size - 1)
    right = np.clip(insert_at, 0, known_at.size - 1)
    distance = np.minimum(np.abs(positions - known_at[left]), np.abs(positions - known_at[right]))
    use = distance <= max_gap
    out[use] = filled[use]
    reached[use] = True
    return out, reached


@dataclass
class ContextStateModel:
    """Профиль плюс пространственно-временной остаток одного коридора."""

    panel: str
    settings: ContextSettings
    profile: WeekdayProfile
    chain: LinkChain
    diagram: DiagramBounds

    def predict_observations(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Вернуть скорость и поток для каждой строки кадра. Метку времени не переписывает."""
        if frame.empty:
            return frame.iloc[0:0].assign(speed_kmh=[], flow_vph=[])
        working = frame.reset_index(drop=True)
        timestamps = working["timestamp"].astype(str)
        link_ids = working["link_id"].astype(str).to_numpy()
        profile_speed, profile_flow = self.profile.predict(link_ids, timestamps)
        speed, flow = self._add_residual(working, timestamps.to_numpy(), link_ids, profile_speed, profile_flow)
        speed, flow = self.diagram.clip(link_ids, speed, flow)
        return pd.DataFrame(
            {
                "timestamp": timestamps.to_numpy(),
                "station_id": working["station_id"].astype(str).to_numpy(),
                "link_id": link_ids,
                "speed_kmh": speed,
                "flow_vph": flow,
            }
        )

    def _add_residual(
        self,
        frame: pd.DataFrame,
        timestamps: np.ndarray,
        link_ids: np.ndarray,
        profile_speed: np.ndarray,
        profile_flow: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        time_labels, time_index = np.unique(timestamps, return_inverse=True)
        n_times = len(time_labels)
        n_links = len(self.chain.link_ids)
        if n_links == 0:
            return profile_speed, profile_flow
        link_position = pd.Series(link_ids).map(self.chain.position)
        on_chain = link_position.notna().to_numpy()
        observed_speed = np.full((n_times, n_links), np.nan)
        observed_flow = np.full((n_times, n_links), np.nan)
        profile_speed_grid = np.zeros((n_times, n_links))
        profile_flow_grid = np.zeros((n_times, n_links))
        weights = observation_weights(frame, self.settings)
        if on_chain.any():
            positions = link_position.to_numpy()[on_chain].astype(np.int64)
            times = time_index[on_chain]
            self._scatter_observed(
                observed_speed,
                observed_flow,
                times,
                positions,
                pd.to_numeric(frame["speed_kmh"], errors="coerce").to_numpy(dtype=float)[on_chain],
                pd.to_numeric(frame["flow_vph"], errors="coerce").to_numpy(dtype=float)[on_chain],
                weights[on_chain],
            )
            self._scatter_mean(profile_speed_grid, times, positions, profile_speed[on_chain])
            self._scatter_mean(profile_flow_grid, times, positions, profile_flow[on_chain])
        speed_grid = self._blend(observed_speed, profile_speed_grid)
        flow_grid = self._blend(observed_flow, profile_flow_grid)
        speed = profile_speed.copy()
        flow = profile_flow.copy()
        if on_chain.any():
            speed[on_chain] = speed_grid[time_index[on_chain], link_position.to_numpy()[on_chain].astype(np.int64)]
            flow[on_chain] = flow_grid[time_index[on_chain], link_position.to_numpy()[on_chain].astype(np.int64)]
        return speed, flow

    def _blend(self, observed: np.ndarray, profile_grid: np.ndarray) -> np.ndarray:
        residual = np.where(np.isfinite(observed), observed - profile_grid, np.nan)
        spatial, spatial_hit = fill_along_axis(residual, self.settings.spatial_hops)
        temporal, temporal_hit = fill_along_axis(residual.T, self.settings.temporal_steps)
        temporal = temporal.T
        temporal_hit = temporal_hit.T
        spatial_mass = np.where(spatial_hit, self.settings.spatial_weight, 0.0)
        temporal_mass = np.where(temporal_hit, self.settings.temporal_weight, 0.0)
        mass = spatial_mass + temporal_mass
        blended = np.zeros(residual.shape, dtype=float)
        np.divide(
            spatial_mass * spatial + temporal_mass * temporal,
            mass,
            out=blended,
            where=mass > 0,
        )
        return profile_grid + blended

    @staticmethod
    def _scatter_observed(
        speed_grid: np.ndarray,
        flow_grid: np.ndarray,
        times: np.ndarray,
        positions: np.ndarray,
        speed: np.ndarray,
        flow: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        speed_sum = np.zeros(speed_grid.shape, dtype=float)
        flow_sum = np.zeros(flow_grid.shape, dtype=float)
        weight_sum = np.zeros(speed_grid.shape, dtype=float)
        usable = np.isfinite(speed) & np.isfinite(flow) & (weights > 0)
        if not usable.any():
            return
        np.add.at(speed_sum, (times[usable], positions[usable]), speed[usable] * weights[usable])
        np.add.at(flow_sum, (times[usable], positions[usable]), flow[usable] * weights[usable])
        np.add.at(weight_sum, (times[usable], positions[usable]), weights[usable])
        np.divide(speed_sum, weight_sum, out=speed_grid, where=weight_sum > 0)
        np.divide(flow_sum, weight_sum, out=flow_grid, where=weight_sum > 0)
        unseen = weight_sum <= 0
        speed_grid[unseen] = np.nan
        flow_grid[unseen] = np.nan

    @staticmethod
    def _scatter_mean(grid: np.ndarray, times: np.ndarray, positions: np.ndarray, values: np.ndarray) -> None:
        # Несколько станций одной связи несут один и тот же профиль. Достаточно последнего значения.
        grid[times, positions] = values


class PanelContextTrainer:
    """Собирает модель одного коридора из немаскированного train и сети."""

    def __init__(self, panel: str, panel_dir: Path, layout: ReleaseLayout, settings: ContextSettings) -> None:
        self.panel = panel
        self.panel_dir = panel_dir
        self.layout = layout
        self.settings = settings

    def fit(self, exclude_dates: set[str], minutes_per_slot: int) -> ContextStateModel:
        chain = LinkChain.read(self.panel_dir)
        diagram = DiagramBounds.read(self.panel_dir, self.settings.speed_epsilon_kmh)
        profile = WeekdayProfile.empty(list(chain.link_ids), minutes_per_slot)
        files = self._train_files()
        if not files:
            raise ProjectError(f"нет немаскированного train у {self.panel}")
        for path in files:
            date_text = _date_from_name(path)
            if date_text is not None and date_text in exclude_dates:
                continue
            frame = pd.read_parquet(
                path,
                columns=["date", "timestamp", "link_id", "speed_kmh", "flow_vph", "pct_observed"],
            )
            if date_text is None:
                frame = frame[~frame["date"].astype(str).isin(exclude_dates)]
            elif "date" in frame.columns and not frame.empty:
                actual = str(frame["date"].iloc[0])
                if actual != date_text:
                    frame = frame[~frame["date"].astype(str).isin(exclude_dates)]
            profile.update(frame, observation_weights(frame, self.settings))
        profile.finalize()
        return ContextStateModel(self.panel, self.settings, profile, chain, diagram)

    def holdout_dates(self) -> set[str]:
        """Даты train, которые профиль проверки не имеет права видеть."""
        held: set[str] = set()
        for path in self._train_files():
            date_text = _date_from_name(path)
            if date_text is None:
                frame = pd.read_parquet(path, columns=["date"])
                date_text = str(frame["date"].iloc[0]) if not frame.empty else ""
            if date_text and self.settings.holds_out(self.panel, date_text):
                held.add(date_text)
        return held

    def masked_files(self, split: str) -> list[Path]:
        directory = self.panel_dir / split / "mainline_states_masked"
        return sorted(directory.glob("**/*.parquet"))

    def _train_files(self) -> list[Path]:
        directory = self.panel_dir / "train" / "mainline_states"
        return sorted(directory.glob("**/*.parquet"))


def _date_from_name(path: Path) -> str | None:
    """Имя файла synthetic_mainline_2030_06_04.parquet хранит календарный день."""
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    year, month, day = parts[-3], parts[-2], parts[-1]
    if len(year) == 4 and year.isdigit() and len(month) == 2 and len(day) == 2:
        return f"{year}-{month}-{day}"
    return None


class StateSubmissionWriter:
    """Сажает прогноз на строки шаблона и сохраняет их метки времени байт в байт."""

    def __init__(self, columns: tuple[str, ...]) -> None:
        self.columns = columns

    def write(self, template_path: Path, predictions: pd.DataFrame, output: Path) -> int:
        template = pd.read_csv(template_path, dtype=str)
        missing = [column for column in ("timestamp", "station_id", "link_id") if column not in template.columns]
        if missing:
            raise ProjectError(f"{template_path} не содержит {missing}")
        forecast = predictions.copy()
        for column in ("timestamp", "station_id", "link_id"):
            forecast[column] = forecast[column].astype(str)
        value_columns = [column for column in ("speed_kmh", "flow_vph") if column in template.columns]
        base = template.drop(columns=value_columns)
        merged = base.merge(forecast, on=["timestamp", "station_id", "link_id"], how="left")
        if merged["speed_kmh"].isna().any() or merged["flow_vph"].isna().any():
            missing_rows = int(merged["speed_kmh"].isna().sum())
            raise ProjectError(f"прогноз не покрыл {missing_rows} строк шаблона {template_path}")
        output.parent.mkdir(parents=True, exist_ok=True)
        ordered = [column for column in self.columns if column in merged.columns]
        merged[ordered].to_csv(output, index=False)
        return len(merged)
