"""Контекст тянет вырезанную связь к живым соседям и не выходит за диаграмму."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trafficflowbench.state.context_model import ContextStateModel  # noqa: E402
from trafficflowbench.state.corridor import DiagramBounds, LinkChain  # noqa: E402
from trafficflowbench.state.profile import WeekdayProfile  # noqa: E402
from trafficflowbench.state.settings import ContextSettings  # noqa: E402


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def model(hops: int, steps: int) -> ContextStateModel:
    settings = ContextSettings(
        temporal_steps=steps,
        spatial_hops=hops,
        spatial_weight=1.0,
        temporal_weight=1.0,
        observation_weight_scale=100,
        minimum_observation_weight=0.05,
        speed_epsilon_kmh=1.0,
        holdout_modulus=7,
        holdout_remainder=0,
        artifact_dir="experiments/artifacts/v2",
    )
    chain = LinkChain.from_topology(
        pd.DataFrame({"link_id": ["A", "B", "C"], "order_index": [0, 1, 2]})
    )
    diagram = DiagramBounds.from_parameters(
        pd.DataFrame(
            {
                "link_id": ["A", "B", "C"],
                "free_speed_kmh": [100, 100, 100],
                "capacity_vph": [2000, 2000, 2000],
                "k_jam": [100, 100, 100],
            }
        ),
        speed_epsilon_kmh=1.0,
    )
    profile = WeekdayProfile.empty(["A", "B", "C"], minutes_per_slot=5)
    history = pd.DataFrame(
        {
            "timestamp": ["2030-06-02T00:00:00Z", "2030-06-02T00:00:00Z", "2030-06-02T00:00:00Z"],
            "link_id": ["A", "B", "C"],
            "speed_kmh": [100, 100, 100],
            "flow_vph": [1000, 1000, 1000],
            "pct_observed": [100, 100, 100],
        }
    )
    from trafficflowbench.state.context_model import observation_weights

    profile.update(history, observation_weights(history, settings))
    profile.finalize()
    return ContextStateModel("TEST", settings, profile, chain, diagram)


def main() -> int:
    settings = ContextSettings.load(REPO / "config" / "context.yaml")
    expect(settings.holds_out("D12_I5_N", "2030-06-01") in (True, False), "хеш дня должен быть булевым")
    same = settings.holds_out("D12_I5_N", "2030-06-01")
    expect(settings.holds_out("D12_I5_N", "2030-06-01") == same, "хеш дня обязан быть устойчивым")

    neighbors = model(hops=2, steps=0)
    day = pd.DataFrame(
        {
            "timestamp": ["2030-06-09T08:00:00Z"] * 3,
            "station_id": ["sa", "sb", "sc"],
            "link_id": ["A", "B", "C"],
            "speed_kmh": [40.0, pd.NA, 40.0],
            "flow_vph": [400.0, pd.NA, 400.0],
            "pct_observed": [100, 0, 100],
        }
    )
    predicted = neighbors.predict_observations(day).set_index("link_id")
    expect(abs(float(predicted.loc["B", "speed_kmh"]) - 40.0) < 1.0, "вырезанная связь должна взять скорость соседей")
    expect(float(predicted.loc["B", "flow_vph"]) <= 2000.0, "поток не выше вместимости")

    isolated = model(hops=0, steps=0)
    kept = isolated.predict_observations(day).set_index("link_id")
    expect(abs(float(kept.loc["B", "speed_kmh"]) - 100.0) < 1e-6, "без соседей в радиусе остаётся профиль")

    over = model(hops=0, steps=0)
    hot = pd.DataFrame(
        {
            "timestamp": ["2030-06-09T08:00:00Z"],
            "station_id": ["sa"],
            "link_id": ["A"],
            "speed_kmh": [150.0],
            "flow_vph": [9000.0],
            "pct_observed": [100],
        }
    )
    bounded = over.predict_observations(hot)
    expect(float(bounded.loc[0, "speed_kmh"]) <= 100.0, "скорость обрезается свободным потоком")
    expect(float(bounded.loc[0, "flow_vph"]) <= 2000.0, "поток обрезается вместимостью")
    print("context checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
