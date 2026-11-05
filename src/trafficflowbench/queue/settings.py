"""Параметры правила очереди. Порог скорости сюда не копируется: он в контракте оценки."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trafficflowbench.config_loader import _as_float, _as_int, _read_mapping
from trafficflowbench.errors import ConfigError


@dataclass(frozen=True)
class QueueSettings:
    """Как сдвигать уже видимую очередь и когда доверять повторяемости train."""

    onset_condition: str
    ongoing_condition: str
    upstream_shift_every_steps: int
    max_upstream_hops: int
    recurrent_share_min: float
    artifact_dir: str

    @classmethod
    def load(cls, path: Path) -> QueueSettings:
        raw = _read_mapping(path)
        settings = cls(
            onset_condition=str(raw.get("onset_condition") or ""),
            ongoing_condition=str(raw.get("ongoing_condition") or ""),
            upstream_shift_every_steps=_as_int(
                raw.get("upstream_shift_every_steps"),
                "upstream_shift_every_steps",
            ),
            max_upstream_hops=_as_int(raw.get("max_upstream_hops"), "max_upstream_hops"),
            recurrent_share_min=_as_float(raw.get("recurrent_share_min"), "recurrent_share_min"),
            artifact_dir=str(raw.get("artifact_dir") or ""),
        )
        if not settings.onset_condition or not settings.ongoing_condition:
            raise ConfigError("имена условий очереди должны быть заданы")
        if settings.onset_condition == settings.ongoing_condition:
            raise ConfigError("onset и ongoing не могут называться одинаково")
        if settings.upstream_shift_every_steps < 0 or settings.max_upstream_hops < 0:
            raise ConfigError("сдвиг очереди вверх по потоку не может быть отрицательным")
        if not 0 < settings.recurrent_share_min <= 1:
            raise ConfigError("recurrent_share_min должен лежать в (0, 1]")
        if not settings.artifact_dir:
            raise ConfigError("queue.artifact_dir должен быть путём")
        return settings

    def hops_after(self, steps_since_birth: int) -> int:
        """Сколько связей выше по потоку добавить через столько шагов после появления очереди."""
        if steps_since_birth <= 0 or self.upstream_shift_every_steps <= 0 or self.max_upstream_hops <= 0:
            return 0
        return min(self.max_upstream_hops, steps_since_birth // self.upstream_shift_every_steps)
