"""Очередь не читает горизонт, сдвигает хвост вверх и не включает onset без повторяемости."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trafficflowbench.queue.forecast import QueueHorizon, RecurrentQueueProfile, latest_speeds  # noqa: E402
from trafficflowbench.queue.settings import QueueSettings  # noqa: E402
from trafficflowbench.state.corridor import LinkChain  # noqa: E402


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def settings() -> QueueSettings:
    return QueueSettings(
        onset_condition="queue_onset",
        ongoing_condition="queue_ongoing",
        upstream_shift_every_steps=2,
        max_upstream_hops=2,
        recurrent_share_min=0.5,
        artifact_dir="experiments/artifacts/v3",
    )


def chain() -> LinkChain:
    return LinkChain.from_topology(pd.DataFrame({"link_id": ["A", "B", "C"], "order_index": [0, 1, 2]}))


def template() -> pd.DataFrame:
    times = [f"2031-03-02T08:{15 + 5 * step:02d}:00Z" for step in range(6)]
    rows = []
    for timestamp in times:
        for link_id in ("A", "B", "C"):
            rows.append({"window_id": "W", "timestamp": timestamp, "link_id": link_id})
    return pd.DataFrame(rows)


def empty_profile() -> RecurrentQueueProfile:
    return RecurrentQueueProfile(("A", "B", "C"), step_minutes=5)


def main() -> int:
    loaded = QueueSettings.load(REPO / "config" / "queue.yaml")
    expect(loaded.max_upstream_hops == 1, "сдвиг по умолчанию — одна связь, не коридор")
    expect(loaded.hops_after(0) == 0, "в момент появления хвост ещё не сдвигается")
    expect(loaded.hops_after(3) == 1, "через три шага добавляется одна связь выше по потоку")

    speeds = latest_speeds(
        pd.DataFrame(
            {
                "timestamp": ["2031-03-02T08:10:00Z", "2031-03-02T08:15:00Z", "2031-03-02T08:20:00Z"],
                "link_id": ["C", "C", "C"],
                "speed_kmh": [40.0, 10.0, 5.0],
                "is_score_eligible": [1, 0, 1],
            }
        ),
        "2031-03-02T08:15:00Z",
    )
    expect(speeds == {"C": 40.0}, "непригодный T и будущий шаг не затирают T-5")
    mixed = latest_speeds(
        pd.DataFrame(
            {
                "timestamp": ["2031-03-02T08:10:00Z", "2031-03-02 08:15:00+00:00"],
                "link_id": ["A", "A"],
                "speed_kmh": [90.0, 30.0],
                "is_score_eligible": [1, 1],
            }
        ),
        "2031-03-02T08:15:00Z",
    )
    expect(mixed == {"A": 30.0}, "оба текстовых вида метки времени читаются, и более поздний T остаётся")

    horizon = QueueHorizon(settings())
    ongoing = horizon.predict(
        "queue_ongoing",
        template(),
        {"A": 100.0, "B": 100.0, "C": 40.0},
        chain(),
        {"A": 60.0, "B": 60.0, "C": 60.0},
        empty_profile(),
    )
    by_time = {
        timestamp: set(group.loc[group["queue_pred"].eq(1), "link_id"])
        for timestamp, group in ongoing.groupby("timestamp", sort=True)
    }
    ordered = sorted(by_time)
    expect(by_time[ordered[0]] == {"C"}, "первые шаги ongoing — только уже стоящая связь")
    expect(by_time[ordered[1]] == {"C"}, "до шага сдвига сосед выше по потоку ещё свободен")
    expect(by_time[ordered[2]] == {"B", "C"}, "через два шага хвост переходит на соседа")
    expect(by_time[ordered[4]] == {"A", "B", "C"}, "дальше хвост доходит до второй связи, не дальше цепочки")
    expect(set(ongoing["timestamp"]) == set(template()["timestamp"]), "метки времени шаблона сохраняются")

    profile = empty_profile()
    profile.seen[0, :, :] = 4
    profile.queued[0, :, :] = 3
    profile.seen[1, :, :] = 4
    profile.queued[1, :, :] = 1
    counted = empty_profile()
    counted.update(
        pd.DataFrame(
            {
                "timestamp": ["2031-03-02T08:15:00Z", "2031-03-02T08:15:00Z"],
                "link_id": ["A", "A"],
                "speed_kmh": [10.0, 100.0],
            }
        ),
        {"A": 60.0, "B": 60.0, "C": 60.0},
    )
    moment = pd.Timestamp("2031-03-02T08:15:00Z")
    slot = (8 * 60 + 15) // 5
    expect(int(counted.seen[0, moment.weekday(), slot]) == 1, "две станции одного момента считаются одним днём")
    expect(int(counted.queued[0, moment.weekday(), slot]) == 1, "очередь есть, если хотя бы одна станция уже ниже порога")
    onset = horizon.predict(
        "queue_onset",
        template(),
        {"A": 100.0, "B": 100.0, "C": 100.0},
        chain(),
        {"A": 60.0, "B": 60.0, "C": 60.0},
        profile,
    )
    onset_links = set(onset.loc[onset["queue_pred"].eq(1), "link_id"])
    expect(onset_links == {"A"}, "onset включает только связь, которая в train стояла чаще, чем нет")
    expect(int(onset.loc[onset["link_id"].eq("C"), "queue_pred"].sum()) == 0, "свободная связь без истории очереди остаётся нулём")

    try:
        horizon.predict("queue_other", template(), {}, chain(), {}, profile)
    except Exception as exc:  # noqa: BLE001 - тест ждёт отказ на чужом условии
        expect("queue_other" in str(exc), "чужое условие должно быть названо в ошибке")
    else:
        raise SystemExit("чужое условие окна обязано отвергаться")
    print("queue checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
