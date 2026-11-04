"""Параметры слоя State v2. Алгоритм получает уже проверенный объект."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from trafficflowbench.config_loader import _as_float, _as_int, _read_mapping
from trafficflowbench.errors import ConfigError


@dataclass(frozen=True)
class ContextSettings:
    """Сколько соседей брать и как взвешивать показания детектора."""

    temporal_steps: int
    spatial_hops: int
    spatial_weight: float
    temporal_weight: float
    observation_weight_scale: float
    minimum_observation_weight: float
    speed_epsilon_kmh: float
    holdout_modulus: int
    holdout_remainder: int
    artifact_dir: str

    @classmethod
    def load(cls, path: Path) -> ContextSettings:
        raw = _read_mapping(path)
        settings = cls(
            temporal_steps=_as_int(raw.get("temporal_steps"), "temporal_steps"),
            spatial_hops=_as_int(raw.get("spatial_hops"), "spatial_hops"),
            spatial_weight=_as_float(raw.get("spatial_weight"), "spatial_weight"),
            temporal_weight=_as_float(raw.get("temporal_weight"), "temporal_weight"),
            observation_weight_scale=_as_float(raw.get("observation_weight_scale"), "observation_weight_scale"),
            minimum_observation_weight=_as_float(
                raw.get("minimum_observation_weight"),
                "minimum_observation_weight",
            ),
            speed_epsilon_kmh=_as_float(raw.get("speed_epsilon_kmh"), "speed_epsilon_kmh"),
            holdout_modulus=_as_int(raw.get("holdout_modulus"), "holdout_modulus"),
            holdout_remainder=_as_int(raw.get("holdout_remainder"), "holdout_remainder"),
            artifact_dir=str(raw.get("artifact_dir") or ""),
        )
        if settings.temporal_steps < 0 or settings.spatial_hops < 0:
            raise ConfigError("temporal_steps и spatial_hops не могут быть отрицательными")
        if settings.spatial_weight < 0 or settings.temporal_weight < 0:
            raise ConfigError("веса контекста не могут быть отрицательными")
        if settings.spatial_weight + settings.temporal_weight <= 0:
            raise ConfigError("хотя бы один вес контекста должен быть положительным")
        if settings.observation_weight_scale <= 0:
            raise ConfigError("observation_weight_scale должен быть положительным")
        if settings.minimum_observation_weight < 0:
            raise ConfigError("minimum_observation_weight не может быть отрицательным")
        if settings.speed_epsilon_kmh <= 0:
            raise ConfigError("speed_epsilon_kmh должен быть положительным")
        if settings.holdout_modulus <= 1:
            raise ConfigError("holdout_modulus должен быть больше 1")
        if not 0 <= settings.holdout_remainder < settings.holdout_modulus:
            raise ConfigError("holdout_remainder должен лежать в диапазоне modulus")
        if not settings.artifact_dir:
            raise ConfigError("context.artifact_dir должен быть путём")
        return settings

    def holds_out(self, panel: str, date_text: str) -> bool:
        """День отложен, если устойчивый хеш панели и даты попадает в заданный остаток.

        Хеш не совпадает с маской Task 1: та решает, какие ячейки вырезаны,
        а этот остаток только делит дни на обучение профиля и проверку.
        """
        digest = hashlib.blake2b(f"{panel}|{date_text}".encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.holdout_modulus == self.holdout_remainder
