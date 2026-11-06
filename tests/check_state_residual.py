"""Минимум соседей не подглядывает в саму ячейку, а S_state следует шкалам контракта."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trafficflowbench.config_loader import ScoringContract  # noqa: E402
from trafficflowbench.state.residual import (  # noqa: E402
    FEATURE_NAMES,
    StateCellScore,
    StateResidualSettings,
    neighbor_min_speed,
    temporal_min_speed,
)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    settings = StateResidualSettings.load(REPO / "config" / "state_residual.yaml")
    expect(settings.min_state_gain > 0, "порог прироста State задан")
    expect(len(FEATURE_NAMES) == 17, "вектор поправки фиксирован")

    grid = np.array([[np.nan, 10.0, 30.0], [80.0, np.nan, 20.0]], dtype=float)
    found = neighbor_min_speed(grid, hops=1)
    expect(found[0, 0] == 10.0 and found[0, 1] == 30.0 and found[0, 2] == 10.0, "минимум берёт соседа, а не себя")
    expect(found[1, 1] == 20.0, "пустая ячейка видит ближайшего конечного соседа")
    over_time = temporal_min_speed(grid, steps=1)
    expect(over_time[0, 2] == 20.0 and over_time[1, 2] == 30.0, "минимум по времени не включает саму пятиминутку")

    scoring = ScoringContract.load(REPO / "config" / "scoring.yaml")
    score = StateCellScore(scoring)
    for regime in scoring.regimes:
        score.add(
            regime,
            np.array([100.0]),
            np.array([1000.0]),
            np.array([75.0]),
            np.array([1000.0]),
            np.array([2.0]),
        )
    # RMSE скорости 25 обнуляет S_speed. Поток совпал, S_flow = 1.
    # S_state = 0.54 * 0 + 0.46 * 1.
    expect(abs(score.regime_value(next(iter(scoring.regimes))) - scoring.state_flow_weight) < 1e-9, "формула режима совпала со шкалой")
    expect(abs(score.value() - scoring.state_flow_weight) < 1e-9, "равное среднее режимов не сдвигает одинаковые режимы")
    print("state residual checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
