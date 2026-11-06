"""Признаки очереди не видят скорость после T, а сдвиг хвоста идёт вверх по потоку."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trafficflowbench.queue.forecast import RecurrentQueueProfile  # noqa: E402
from trafficflowbench.queue.model_settings import QueueModelSettings  # noqa: E402
from trafficflowbench.queue.settings import QueueSettings  # noqa: E402
from trafficflowbench.queue.supervised import DayLattice, ObservedQueueRule, origin_batch  # noqa: E402


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def lattice() -> DayLattice:
    times = pd.date_range("2031-03-02T00:00:00Z", periods=40, freq="5min", tz="UTC")
    speed = np.full((3, 40), 100.0)
    flow = np.full((3, 40), 500.0)
    occupancy = np.full((3, 40), 0.05)
    free = np.array([100.0, 100.0, 100.0])
    capacity = np.array([2000.0, 2000.0, 2000.0])
    thresholds = free * 0.6
    return DayLattice(("A", "B", "C"), pd.DatetimeIndex(times), speed, flow, occupancy, free, capacity, thresholds)


def main() -> int:
    loaded = QueueModelSettings.load(REPO / "config" / "queue_model.yaml")
    expect(loaded.holds_out("D12_I5_N", "2030-06-01") == loaded.holds_out("D12_I5_N", "2030-06-01"), "хеш дня устойчив")
    expect(0 < loaded.min_iou_gain < 1, "запас IoU задан")

    profile = RecurrentQueueProfile(("A", "B", "C"), step_minutes=5)
    before, labels_before, _ = origin_batch(lattice(), origin=20, history_points=13, horizon=6, short_steps=3, profile=profile)
    changed = lattice()
    changed.speed[:, 21:] = 0.0
    after, labels_after, valid = origin_batch(changed, origin=20, history_points=13, horizon=6, short_steps=3, profile=profile)
    expect(before.shape[1] == 16, "вектор признаков фиксирован")
    expect(np.allclose(before, after, equal_nan=True), "скорость после T не меняет признаки")
    expect(not labels_before.any(), "свободный поток не размечается очередью")
    expect(bool(labels_after[valid].all()), "нулевая скорость горизонта становится меткой очереди")

    rule = ObservedQueueRule(
        QueueSettings(
            onset_condition="queue_onset",
            ongoing_condition="queue_ongoing",
            upstream_shift_every_steps=2,
            max_upstream_hops=2,
            recurrent_share_min=0.5,
            artifact_dir="experiments/artifacts/v3",
        )
    )
    shares = np.zeros((3, 6))
    predicted = rule.predict(np.array([False, False, True]), shares, onset=False)
    expect(set(np.flatnonzero(predicted[:, 0])) == {2}, "в момент появления очередь стоит только на своей связи")
    expect(set(np.flatnonzero(predicted[:, 2])) == {1, 2}, "через два шага хвост переходит на соседа выше по потоку")
    expect(set(np.flatnonzero(predicted[:, 4])) == {0, 1, 2}, "дальше хвост доходит до начала цепочки")
    onset = rule.predict(np.array([False, False, False]), np.array([[0.8, 0.8, 0.8, 0.8, 0.8, 0.8], [0.1] * 6, [0.1] * 6]), onset=True)
    expect(set(np.flatnonzero(onset[:, 0])) == {0}, "onset без наблюдения включает только повторяемую связь")
    print("queue model checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
