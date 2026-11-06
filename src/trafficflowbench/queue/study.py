"""Прогон обучения по одному коридору и решение, заменять ли принятое правило."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trafficflowbench.config_loader import ScoringContract
from trafficflowbench.errors import ProjectError
from trafficflowbench.queue.forecast import RecurrentQueueProfile
from trafficflowbench.queue.model_settings import QueueModelSettings
from trafficflowbench.queue.settings import QueueSettings
from trafficflowbench.queue.supervised import (
    BalancedDraw,
    DayLattice,
    ObservedQueueRule,
    ProxyWindow,
    SplitQueueModel,
    TrainedQueueModel,
    equal_mean,
    feature_matrix,
    origin_batch,
    share_matrix,
)
from trafficflowbench.state.corridor import LinkChain

DAY_COLUMNS = ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "is_score_eligible"]
HISTORY_COLUMNS = ["window_id", "timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "is_score_eligible"]


@dataclass
class QueueFit:
    """Модель одного коридора и выбор: оставить правило, заменить его или смешать."""

    panel: str
    kind: str
    threshold: float
    model: TrainedQueueModel | SplitQueueModel | None
    profile: RecurrentQueueProfile | None
    chain: LinkChain
    free_speed: np.ndarray
    capacity: np.ndarray
    thresholds: np.ndarray
    threshold_by_link: dict[str, float]
    summary: dict[str, float | int | str]


class QueuePanelStudy:
    """Учит коридор на train без отложенных дней и сверяет IoU с правилом v3."""

    def __init__(self, panel: str, release_root: Path, scoring: ScoringContract, model: QueueModelSettings, rule: QueueSettings) -> None:
        self.panel = panel
        self.release_root = release_root
        self.scoring = scoring
        self.model_settings = model
        self.rule = rule
        self.panel_dir = release_root / "corridors" / panel
        self.chain = LinkChain.read(self.panel_dir)
        self.free_speed, self.capacity, self.thresholds, self.threshold_by_link = _limits(
            self.panel_dir,
            self.chain,
            scoring.queue_free_speed_fraction,
        )
        self.step_minutes = scoring.queue_step_minutes
        self.history_points = _points(scoring.queue_history_minutes, self.step_minutes, include_origin=True)
        self.horizon_steps = _points(scoring.queue_horizon_minutes, self.step_minutes, include_origin=False)
        self.stride_steps = _points(model.train_stride_minutes, self.step_minutes, include_origin=False)
        self.separation_steps = _points(model.eval_separation_minutes, self.step_minutes, include_origin=False)
        if self.model_settings.short_steps >= self.history_points:
            raise ProjectError("short_steps должен быть короче истории до T")

    def fit(self, previous_score: float | None = None) -> QueueFit:
        files = sorted((self.panel_dir / "train" / "mainline_states").glob("**/*.parquet"))
        if not files:
            raise ProjectError(f"нет train для очереди: {self.panel}")
        dated = [(path, _date_from_path(path)) for path in files]
        if any(date is None for _, date in dated):
            raise ProjectError(f"не у всех дней {self.panel} дата читается из имени файла")
        holdout = sorted(date for _, date in dated if date and self.model_settings.holds_out(self.panel, date))
        calibration = set(holdout[0::2])
        report = set(holdout[1::2])
        train_paths = [path for path, date in dated if date not in set(holdout)]
        profile = RecurrentQueueProfile(self.chain.link_ids, self.step_minutes)
        for path in train_paths:
            profile.update(pd.read_parquet(path, columns=["timestamp", "link_id", "speed_kmh"]), self.threshold_by_link)
        seed = self.model_settings.random_state
        rng = np.random.default_rng(seed)
        extended = self.model_settings.separate_conditions
        draw = BalancedDraw(self.model_settings.max_positive_rows, self.model_settings.max_negative_rows, np.random.default_rng(seed))
        draw_onset = BalancedDraw(
            self.model_settings.max_positive_rows,
            self.model_settings.max_negative_rows,
            np.random.default_rng(seed + 1),
        )
        draw_ongoing = BalancedDraw(
            self.model_settings.max_positive_rows,
            self.model_settings.max_negative_rows,
            np.random.default_rng(seed + 2),
        )
        for path, date in dated:
            if date in set(holdout):
                continue
            lattice = self._lattice(path)
            if lattice is None:
                continue
            self._add_training_day(lattice, profile, draw, draw_onset, draw_ongoing, rng, extended)
        trained = _fit_draw(draw, self.model_settings)
        split_model = _fit_split(draw_onset, draw_ongoing, trained, self.model_settings) if extended else None
        windows = self._holdout_windows(dated, holdout, calibration, profile, trained, split_model, extended)
        choice, threshold, summary = self._choose(
            trained,
            split_model,
            windows,
            calibration,
            report,
            draw,
            previous_score,
        )
        shipped: TrainedQueueModel | SplitQueueModel | None
        if choice == "split" and split_model is not None:
            shipped = split_model
            profile_kept = profile
        elif choice in {"model", "blend"} and trained is not None:
            trained.threshold = threshold
            shipped = trained
            profile_kept = profile
        else:
            shipped = None
            profile_kept = None
        summary["panel"] = self.panel
        summary["kind"] = choice
        return QueueFit(
            self.panel,
            choice,
            threshold,
            shipped,
            profile_kept,
            self.chain,
            self.free_speed,
            self.capacity,
            self.thresholds,
            self.threshold_by_link,
            summary,
        )

    def predict_split(self, fit: QueueFit, split: str) -> pd.DataFrame:
        """Строки шаблона, которые выбранная модель заменяет. Правило сюда не входит."""
        if fit.kind in {"rule", "keep"} or fit.model is None or fit.profile is None:
            return pd.DataFrame(columns=["window_id", "timestamp", "link_id", "queue_pred"])
        task_dir = self.release_root / "task2" / self.panel / split
        windows = pd.read_csv(task_dir / "window_index.csv", dtype=str)
        template = pd.read_csv(task_dir / "sample_submission_queue.csv", dtype=str)
        template["window_id"] = template["window_id"].astype(str)
        history = pd.read_parquet(task_dir / "window_history.parquet", columns=HISTORY_COLUMNS)
        history["window_id"] = history["window_id"].astype(str)
        origins = _origin_frames(self.panel_dir / split / "mainline_states_masked", set(pd.to_datetime(windows["forecast_origin"], utc=True)))
        pieces: list[pd.DataFrame] = []
        for window_id, part in template.groupby("window_id", sort=False):
            info = windows[windows["window_id"].astype(str) == str(window_id)]
            if info.empty:
                raise ProjectError(f"окно {window_id} есть в шаблоне и нет в индексе")
            condition = str(info["condition"].iloc[0])
            if not _uses_model(fit.kind, condition, self.rule.onset_condition):
                continue
            origin = pd.to_datetime(info["forecast_origin"].iloc[0], utc=True)
            observed = history[history["window_id"] == str(window_id)]
            extra = origins.get(origin)
            if extra is not None and not extra.empty:
                observed = pd.concat([observed, extra], ignore_index=True)
            pieces.append(self._predict_window(fit, part, observed, origin, condition == self.rule.onset_condition))
        if not pieces:
            return pd.DataFrame(columns=["window_id", "timestamp", "link_id", "queue_pred"])
        return pd.concat(pieces, ignore_index=True)

    def _predict_window(
        self,
        fit: QueueFit,
        template: pd.DataFrame,
        observed: pd.DataFrame,
        origin: pd.Timestamp,
        onset: bool,
    ) -> pd.DataFrame:
        speed, flow, occupancy = _history_block(
            observed,
            self.chain,
            origin,
            self.step_minutes,
            self.history_points,
        )
        order = sorted(pd.to_datetime(template["timestamp"], utc=True, format="ISO8601").unique())
        if len(order) != self.horizon_steps:
            raise ProjectError(f"в окне {template['window_id'].iloc[0]} не {self.horizon_steps} шагов горизонта")
        expected = origin + pd.Timedelta(minutes=self.step_minutes)
        if order[0] != expected:
            raise ProjectError(f"горизонт окна {template['window_id'].iloc[0]} начинается не в T+шаг")
        future = pd.DatetimeIndex(order)
        shares = share_matrix(fit.profile, future)
        shares_now = share_matrix(fit.profile, pd.DatetimeIndex([origin]))[:, 0]
        view = DayLattice(
            self.chain.link_ids,
            future,
            speed,
            flow,
            occupancy,
            self.free_speed,
            self.capacity,
            self.thresholds,
        )
        extended = self.model_settings.separate_conditions
        features = feature_matrix(
            speed,
            flow,
            occupancy,
            shares,
            future,
            view,
            self.model_settings.short_steps,
            shares_now,
            extended,
        )
        probability, cut = _window_probability(fit, features, onset)
        expected = len(self.chain.link_ids) * self.horizon_steps
        if probability.shape != (expected,):
            raise ProjectError(
                f"модель вернула {probability.shape[0]} вероятностей, окно ждёт {expected}"
            )
        chosen = (probability >= cut).reshape(len(self.chain.link_ids), self.horizon_steps)
        column_of = {stamp: index for index, stamp in enumerate(order)}
        predicted: list[int] = []
        for timestamp, link_id in zip(template["timestamp"], template["link_id"].astype(str), strict=True):
            row = self.chain.position.get(link_id)
            column = column_of.get(pd.to_datetime(timestamp, utc=True, format="ISO8601"))
            if row is None or column is None:
                predicted.append(0)
            else:
                predicted.append(int(chosen[row, column]))
        result = template[["window_id", "timestamp", "link_id"]].copy()
        result["queue_pred"] = predicted
        return result

    def _lattice(self, path: Path) -> DayLattice | None:
        frame = pd.read_parquet(path, columns=DAY_COLUMNS)
        return DayLattice.from_frame(frame, self.chain.link_ids, self.free_speed, self.capacity, self.thresholds)

    def _add_training_day(
        self,
        lattice: DayLattice,
        profile: RecurrentQueueProfile,
        draw: BalancedDraw,
        draw_onset: BalancedDraw,
        draw_ongoing: BalancedDraw,
        rng: np.random.Generator,
        extended: bool,
    ) -> None:
        times = lattice.times
        last = lattice.speed.shape[1] - self.horizon_steps - 1
        for origin in range(self.history_points - 1, last + 1, self.stride_steps):
            if not _contiguous(times, origin - self.history_points + 1, origin, self.step_minutes):
                continue
            if not _contiguous(times, origin + 1, origin + self.horizon_steps, self.step_minutes):
                continue
            future = lattice.speed[:, origin + 1 : origin + 1 + self.horizon_steps]
            history = lattice.speed[:, origin - self.history_points + 1 : origin + 1]
            history_queued = bool(_queued(history, lattice.thresholds).any())
            interesting = _queued(future, lattice.thresholds).any() or history_queued
            if not interesting and float(rng.random()) > self.model_settings.quiet_keep:
                continue
            features, labels, valid = origin_batch(
                lattice,
                origin,
                self.history_points,
                self.horizon_steps,
                self.model_settings.short_steps,
                profile,
                extended=extended,
            )
            draw.add(features[valid], labels[valid])
            if not extended:
                continue
            # Окно соревнования целиком onset или ongoing: деление такое же.
            specialist = draw_ongoing if history_queued else draw_onset
            specialist.add(features[valid], labels[valid])

    def _holdout_windows(
        self,
        dated: list[tuple[Path, str | None]],
        holdout: list[str],
        calibration: set[str],
        profile: RecurrentQueueProfile,
        trained: TrainedQueueModel | None,
        split_model: SplitQueueModel | None,
        extended: bool,
    ) -> list[ProxyWindow]:
        if trained is None:
            return []
        rule = ObservedQueueRule(self.rule)
        windows: list[ProxyWindow] = []
        holdout_set = set(holdout)
        for path, date in dated:
            if date not in holdout_set or date is None:
                continue
            lattice = self._lattice(path)
            if lattice is None:
                continue
            tag = "calibration" if date in calibration else "report"
            windows.extend(self._windows_of_day(lattice, profile, trained, split_model, rule, tag, extended))
        return windows

    def _windows_of_day(
        self,
        lattice: DayLattice,
        profile: RecurrentQueueProfile,
        trained: TrainedQueueModel,
        split_model: SplitQueueModel | None,
        rule: ObservedQueueRule,
        tag: str,
        extended: bool,
    ) -> list[ProxyWindow]:
        found: list[ProxyWindow] = []
        times = lattice.times
        next_origin = self.history_points - 1
        last = lattice.speed.shape[1] - self.horizon_steps - 1
        for origin in range(self.history_points - 1, last + 1):
            if origin < next_origin:
                continue
            if not _contiguous(times, origin - self.history_points + 1, origin, self.step_minutes):
                continue
            if not _contiguous(times, origin + 1, origin + self.horizon_steps, self.step_minutes):
                continue
            future = lattice.speed[:, origin + 1 : origin + 1 + self.horizon_steps]
            truth = _queued(future, lattice.thresholds)
            if not truth.any():
                continue
            history = lattice.speed[:, origin - self.history_points + 1 : origin + 1]
            onset = not bool(_queued(history, lattice.thresholds).any())
            observed = _queued(lattice.speed[:, origin], lattice.thresholds)
            features, labels, valid = origin_batch(
                lattice,
                origin,
                self.history_points,
                self.horizon_steps,
                self.model_settings.short_steps,
                profile,
                extended=extended,
            )
            shares = share_matrix(profile, times[origin + 1 : origin + 1 + self.horizon_steps])
            ruled = rule.predict(observed, shares, onset=onset).reshape(-1)
            split_probability = None
            if split_model is not None and split_model.ready():
                split_probability = split_model.probabilities(features, onset=onset)[valid]
            found.append(
                ProxyWindow(
                    split=tag,
                    condition=self.rule.onset_condition if onset else self.rule.ongoing_condition,
                    truth=labels[valid],
                    persistence=np.repeat(observed, self.horizon_steps)[valid],
                    rule=ruled[valid],
                    probability=trained.probabilities(features)[valid],
                    split_probability=split_probability,
                )
            )
            next_origin = origin + self.separation_steps
        return found

    def _choose(
        self,
        trained: TrainedQueueModel | None,
        split_model: SplitQueueModel | None,
        windows: list[ProxyWindow],
        calibration: set[str],
        report: set[str],
        draw: BalancedDraw,
        previous_score: float | None,
    ) -> tuple[str, float, dict[str, float | int | str]]:
        conditions = (self.rule.onset_condition, self.rule.ongoing_condition)
        report_windows = [window for window in windows if window.split == "report"]
        calibration_windows = [window for window in windows if window.split == "calibration"]
        threshold = 0.5
        if trained is not None and calibration_windows:
            best_score = -1.0
            for candidate in self.model_settings.decision_thresholds:
                score = equal_mean(
                    calibration_windows,
                    lambda window, cut=candidate: window.probability >= cut,
                    self.model_settings.min_windows,
                    conditions,
                )
                if score is not None and score > best_score:
                    best_score = score
                    threshold = candidate
        onset_cut, ongoing_cut = _split_thresholds(split_model, calibration_windows, self.model_settings, conditions)
        if split_model is not None:
            split_model.onset_threshold = onset_cut
            split_model.ongoing_threshold = ongoing_cut
        summary: dict[str, float | int | str] = {
            "threshold": threshold,
            "onset_threshold": onset_cut,
            "ongoing_threshold": ongoing_cut,
            "train_positive": draw.positive_rows,
            "train_negative": draw.negative_rows,
            "report_onset": sum(window.condition == self.rule.onset_condition for window in report_windows),
            "report_ongoing": sum(window.condition == self.rule.ongoing_condition for window in report_windows),
        }
        if trained is None:
            summary["report_persistence"] = ""
            summary["report_rule"] = ""
            summary["report_model"] = ""
            summary["report_blend"] = ""
            summary["report_split"] = ""
            return "rule", threshold, summary

        def model_bits(window: ProxyWindow, cut: float = threshold) -> np.ndarray:
            return window.probability >= cut

        def blend_bits(window: ProxyWindow) -> np.ndarray:
            if window.condition == self.rule.onset_condition:
                return model_bits(window)
            return window.rule

        def split_bits(window: ProxyWindow) -> np.ndarray:
            if window.split_probability is None:
                return model_bits(window)
            cut = onset_cut if window.condition == self.rule.onset_condition else ongoing_cut
            return window.split_probability >= cut

        scores = {
            "report_persistence": equal_mean(report_windows, lambda window: window.persistence, self.model_settings.min_windows, conditions),
            "report_rule": equal_mean(report_windows, lambda window: window.rule, self.model_settings.min_windows, conditions),
            "report_model": equal_mean(report_windows, model_bits, self.model_settings.min_windows, conditions),
            "report_blend": equal_mean(report_windows, blend_bits, self.model_settings.min_windows, conditions),
            "report_split": equal_mean(report_windows, split_bits, self.model_settings.min_windows, conditions),
        }
        for name, score in scores.items():
            summary[name] = "" if score is None else round(score, 6)
        rule_score = scores["report_rule"]
        onset_n = int(summary["report_onset"])
        ongoing_n = int(summary["report_ongoing"])
        choice = "rule"
        chosen_score: float | None = None
        if rule_score is not None:
            # Полная замена действует и на ongoing. Её можно брать, только когда оба типа окон есть.
            candidates = (
                ("model", scores["report_model"], onset_n >= self.model_settings.min_windows and ongoing_n >= self.model_settings.min_windows),
                ("blend", scores["report_blend"], onset_n >= self.model_settings.min_windows),
                ("split", scores["report_split"], split_model is not None and split_model.ready() and onset_n >= self.model_settings.min_windows and ongoing_n >= self.model_settings.min_windows),
            )
            for name, score, allowed in candidates:
                if not allowed or score is None or score < rule_score + self.model_settings.min_iou_gain:
                    continue
                if chosen_score is None or score >= chosen_score:
                    choice = name
                    chosen_score = score
        if previous_score is not None and (chosen_score is None or chosen_score < previous_score + self.model_settings.min_iou_gain):
            choice = "keep"
        return choice, threshold, summary


def _fit_draw(draw: BalancedDraw, settings: QueueModelSettings) -> TrainedQueueModel | None:
    packed = draw.finish()
    if packed is None or int(packed[1].sum()) < settings.min_positive_rows:
        return None
    return TrainedQueueModel.fit(packed[0], packed[1], settings, threshold=0.5)


def _fit_split(
    draw_onset: BalancedDraw,
    draw_ongoing: BalancedDraw,
    fallback: TrainedQueueModel | None,
    settings: QueueModelSettings,
) -> SplitQueueModel | None:
    onset = _fit_draw(draw_onset, settings)
    ongoing = _fit_draw(draw_ongoing, settings)
    if onset is None and ongoing is None:
        return None
    return SplitQueueModel(onset, ongoing, fallback, onset_threshold=0.5, ongoing_threshold=0.5)


def _split_thresholds(
    split_model: SplitQueueModel | None,
    calibration: list[ProxyWindow],
    settings: QueueModelSettings,
    conditions: tuple[str, str],
) -> tuple[float, float]:
    """Независимые пороги onset и ongoing на калибровочной половине отложенных дней."""
    if split_model is None or not calibration or not any(window.split_probability is not None for window in calibration):
        return 0.5, 0.5
    best = -1.0
    onset_cut = 0.5
    ongoing_cut = 0.5
    for onset_candidate in settings.decision_thresholds:
        for ongoing_candidate in settings.decision_thresholds:
            def bits(
                window: ProxyWindow,
                onset_value: float = onset_candidate,
                ongoing_value: float = ongoing_candidate,
            ) -> np.ndarray:
                if window.split_probability is None:
                    return window.probability >= onset_value
                cut = onset_value if window.condition == conditions[0] else ongoing_value
                return window.split_probability >= cut

            score = equal_mean(calibration, bits, settings.min_windows, conditions)
            if score is not None and score > best:
                best = score
                onset_cut = onset_candidate
                ongoing_cut = ongoing_candidate
    return onset_cut, ongoing_cut


def _window_probability(fit: QueueFit, features: np.ndarray, onset: bool) -> tuple[np.ndarray, float]:
    if isinstance(fit.model, SplitQueueModel):
        return fit.model.probabilities(features, onset=onset), fit.model.threshold_for(onset)
    if fit.model is None:
        return np.zeros(len(features), dtype=float), fit.threshold
    return fit.model.probabilities(features), fit.threshold


def _uses_model(kind: str, condition: str, onset_condition: str) -> bool:
    if kind in {"model", "split"}:
        return True
    if kind == "blend":
        return condition == onset_condition
    return False


def _queued(speed: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    limit = thresholds if speed.ndim == 1 else thresholds[:, None]
    return np.isfinite(speed) & (speed <= limit)


def _contiguous(times: pd.DatetimeIndex, start: int, end: int, step_minutes: int) -> bool:
    if start < 0 or end >= len(times) or end < start:
        return False
    return times[end] - times[start] == pd.Timedelta(minutes=(end - start) * step_minutes)


def _points(minutes: int, step_minutes: int, include_origin: bool) -> int:
    if step_minutes <= 0 or minutes <= 0 or minutes % step_minutes != 0:
        raise ProjectError("длительность очереди должна делиться на шаг без остатка")
    points = minutes // step_minutes
    return points + 1 if include_origin else points


def _limits(panel_dir: Path, chain: LinkChain, fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    frame = pd.read_csv(panel_dir / "network" / "fd_parameters.csv")
    frame["link_id"] = frame["link_id"].astype(str)
    free = dict(zip(frame["link_id"], pd.to_numeric(frame["free_speed_kmh"], errors="coerce")))
    capacity = dict(zip(frame["link_id"], pd.to_numeric(frame["capacity_vph"], errors="coerce")))
    free_speed = np.array([free.get(link_id, np.nan) for link_id in chain.link_ids], dtype=float)
    capacity_vph = np.array([capacity.get(link_id, np.nan) for link_id in chain.link_ids], dtype=float)
    thresholds = free_speed * fraction
    by_link = {
        link_id: float(thresholds[index])
        for index, link_id in enumerate(chain.link_ids)
        if np.isfinite(thresholds[index]) and thresholds[index] > 0
    }
    return free_speed, capacity_vph, thresholds, by_link


def _date_from_path(path: Path) -> str | None:
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    year, month, day = parts[-3], parts[-2], parts[-1]
    if len(year) == 4 and year.isdigit() and len(month) == 2 and len(day) == 2:
        return f"{year}-{month}-{day}"
    return None


def _history_block(
    frame: pd.DataFrame,
    chain: LinkChain,
    origin: pd.Timestamp,
    step_minutes: int,
    points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Скорость, поток и занятость на сетке [T-история, T]. Будущее отбрасывается."""
    width = len(chain.link_ids)
    speed = np.full((width, points), np.nan)
    flow = np.full((width, points), np.nan)
    occupancy = np.full((width, points), np.nan)
    if frame.empty:
        return speed, flow, occupancy
    times = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    eligible = pd.to_numeric(frame["is_score_eligible"], errors="coerce").fillna(0.0).to_numpy() >= 0.5
    steps_back = ((origin - times).dt.total_seconds() / 60.0 / step_minutes).round()
    steps_np = steps_back.to_numpy(dtype=float)
    keep = eligible & np.isfinite(steps_np) & (steps_np >= 0) & (steps_np < points) & (times <= origin).to_numpy()
    if not keep.any():
        return speed, flow, occupancy
    columns = (points - 1 - steps_np[keep]).astype(np.int64)
    link_index = frame["link_id"].astype(str).map(chain.position)
    rows = link_index.to_numpy()[keep]
    known = pd.notna(rows)
    if not np.asarray(known).any():
        return speed, flow, occupancy
    known_np = np.asarray(known)
    rows = rows[known_np].astype(np.int64)
    columns = columns[known_np]
    values = pd.to_numeric(frame["speed_kmh"], errors="coerce").to_numpy()[keep][known_np]
    finite = np.isfinite(values)
    grid = np.full((width, points), np.inf)
    np.minimum.at(grid, (rows[finite], columns[finite]), values[finite])
    speed = np.where(np.isfinite(grid), grid, np.nan)
    for target, source in ((flow, "flow_vph"), (occupancy, "occupancy")):
        total = np.zeros((width, points))
        count = np.zeros((width, points))
        raw = pd.to_numeric(frame[source], errors="coerce").to_numpy()[keep][known_np]
        usable = np.isfinite(raw)
        np.add.at(total, (rows[usable], columns[usable]), raw[usable])
        np.add.at(count, (rows[usable], columns[usable]), 1.0)
        np.divide(total, count, out=target, where=count > 0)
    return speed, flow, occupancy


def _origin_frames(directory: Path, needed: set[pd.Timestamp]) -> dict[pd.Timestamp, pd.DataFrame]:
    files = sorted(directory.glob("**/*.parquet")) if directory.is_dir() else []
    pieces: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_parquet(path, columns=DAY_COLUMNS)
        times = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
        keep = times.isin(needed)
        if not bool(keep.any()):
            continue
        piece = frame.loc[keep, ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "is_score_eligible"]].copy()
        piece["_key"] = times.loc[keep].to_numpy()
        pieces.append(piece)
    if not pieces:
        return {}
    stacked = pd.concat(pieces, ignore_index=True)
    return {
        pd.Timestamp(origin): group.drop(columns=["_key"])
        for origin, group in stacked.groupby("_key", sort=False)
    }
