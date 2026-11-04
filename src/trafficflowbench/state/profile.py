"""Профиль скорости и потока: день недели и слот пятиминутки, отдельно на каждую связь.

Вклад ячейки равен её весу доверия. Ряд с низкой долей наблюдения остаётся в сумме,
а не выбрасывается порогом пригодности к оценке.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WeekdayProfile:
    """Накапливает взвешенные средние и отдаёт прогноз по связи и метке времени."""

    minutes_per_slot: int
    slots_per_day: int
    link_index: dict[str, int]
    speed_sum: np.ndarray
    speed_weight: np.ndarray
    flow_sum: np.ndarray
    flow_weight: np.ndarray
    speed_mean: np.ndarray | None = None
    flow_mean: np.ndarray | None = None
    speed_fallback: np.ndarray | None = None
    flow_fallback: np.ndarray | None = None
    speed_global: float = 0.0
    flow_global: float = 0.0

    @classmethod
    def empty(cls, link_ids: list[str], minutes_per_slot: int) -> WeekdayProfile:
        if minutes_per_slot <= 0 or 24 * 60 % minutes_per_slot != 0:
            raise ValueError("минуты слота должны нацело делить сутки")
        slots_per_day = 24 * 60 // minutes_per_slot
        n_slots = 7 * slots_per_day
        n_links = len(link_ids)
        zeros = np.zeros((n_links, n_slots), dtype=np.float64)
        return cls(
            minutes_per_slot=minutes_per_slot,
            slots_per_day=slots_per_day,
            link_index={link_id: index for index, link_id in enumerate(link_ids)},
            speed_sum=zeros.copy(),
            speed_weight=zeros.copy(),
            flow_sum=zeros.copy(),
            flow_weight=zeros.copy(),
        )

    def update(self, frame: pd.DataFrame, weights: np.ndarray) -> None:
        """Добавить один кадр наблюдений. Нулевые веса и пустые скорости в среднее не входят."""
        if frame.empty:
            return
        link_ids = frame["link_id"].astype(str)
        known = link_ids.map(self.link_index)
        mask = known.notna().to_numpy()
        if not mask.any():
            return
        indexes = known.to_numpy()[mask].astype(np.int64)
        slots = self.slots(frame["timestamp"])[mask]
        speed = pd.to_numeric(frame["speed_kmh"], errors="coerce").to_numpy(dtype=float)[mask]
        flow = pd.to_numeric(frame["flow_vph"], errors="coerce").to_numpy(dtype=float)[mask]
        weight = np.asarray(weights, dtype=float)[mask]
        self._accumulate(self.speed_sum, self.speed_weight, indexes, slots, speed, weight)
        self._accumulate(self.flow_sum, self.flow_weight, indexes, slots, flow, weight)

    def finalize(self) -> None:
        """Поделить накопленные суммы. Пустой слот берёт среднее связи, пустая связь — среднее коридора."""
        self.speed_mean = self._divide(self.speed_sum, self.speed_weight)
        self.flow_mean = self._divide(self.flow_sum, self.flow_weight)
        self.speed_fallback = self._fallback(self.speed_sum, self.speed_weight)
        self.flow_fallback = self._fallback(self.flow_sum, self.flow_weight)
        speed_total = float(self.speed_weight.sum())
        flow_total = float(self.flow_weight.sum())
        self.speed_global = float(self.speed_sum.sum() / speed_total) if speed_total > 0 else 0.0
        self.flow_global = float(self.flow_sum.sum() / flow_total) if flow_total > 0 else 0.0

    def predict(self, link_ids: np.ndarray, timestamps: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Прогноз той же длины, что и вход. Неизвестная связь получает среднее коридора."""
        if self.speed_mean is None or self.flow_mean is None:
            raise RuntimeError("профиль ещё не собран")
        slots = self.slots(timestamps)
        speed = np.full(len(link_ids), self.speed_global, dtype=float)
        flow = np.full(len(link_ids), self.flow_global, dtype=float)
        indexes = pd.Series(link_ids).map(self.link_index)
        known = indexes.notna().to_numpy()
        if not known.any():
            return speed, flow
        li = indexes.to_numpy()[known].astype(np.int64)
        sl = slots[known]
        speed_slot = self.speed_mean[li, sl]
        flow_slot = self.flow_mean[li, sl]
        speed_empty = self.speed_weight[li, sl] <= 0
        flow_empty = self.flow_weight[li, sl] <= 0
        speed[known] = np.where(speed_empty, self.speed_fallback[li], speed_slot)
        flow[known] = np.where(flow_empty, self.flow_fallback[li], flow_slot)
        return speed, flow

    def slots(self, timestamps: pd.Series) -> np.ndarray:
        """Слот недели: семь дней по числу пятиминуток в сутках. Время читается как UTC."""
        clock = pd.to_datetime(timestamps, utc=True)
        minute_of_day = clock.dt.hour.to_numpy(dtype=np.int64) * 60 + clock.dt.minute.to_numpy(dtype=np.int64)
        slot_in_day = minute_of_day // self.minutes_per_slot
        weekday = clock.dt.weekday.to_numpy(dtype=np.int64)
        return weekday * self.slots_per_day + slot_in_day

    @staticmethod
    def _accumulate(
        total: np.ndarray,
        weight_total: np.ndarray,
        indexes: np.ndarray,
        slots: np.ndarray,
        values: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        usable = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not usable.any():
            return
        np.add.at(total, (indexes[usable], slots[usable]), values[usable] * weights[usable])
        np.add.at(weight_total, (indexes[usable], slots[usable]), weights[usable])

    @staticmethod
    def _divide(total: np.ndarray, weight: np.ndarray) -> np.ndarray:
        mean = np.zeros_like(total)
        np.divide(total, weight, out=mean, where=weight > 0)
        return mean

    @staticmethod
    def _fallback(total: np.ndarray, weight: np.ndarray) -> np.ndarray:
        summed = total.sum(axis=1)
        denom = weight.sum(axis=1)
        fallback = np.zeros(total.shape[0], dtype=float)
        np.divide(summed, denom, out=fallback, where=denom > 0)
        return fallback
