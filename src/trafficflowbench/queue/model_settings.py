"""Параметры обучения очереди. Порог скорости сюда не копируется."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from trafficflowbench.config_loader import _as_float, _as_int, _read_mapping
from trafficflowbench.errors import ConfigError


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{label} должен быть true или false")
    return value


def _as_float_list(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{label} должен быть непустым списком чисел")
    return tuple(_as_float(item, label) for item in value)


@dataclass(frozen=True)
class QueueModelSettings:
    """Как учить очередь и когда заменять уже принятое правило."""

    holdout_modulus: int
    holdout_remainder: int
    train_stride_minutes: int
    eval_separation_minutes: int
    short_steps: int
    quiet_keep: float
    max_positive_rows: int
    max_negative_rows: int
    min_positive_rows: int
    min_windows: int
    min_iou_gain: float
    decision_thresholds: tuple[float, ...]
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    random_state: int
    artifact_dir: str
    # Отдельные модели onset и ongoing. Пустые строки значат «как в прежнем прогоне».
    separate_conditions: bool
    base_artifact: str
    previous_report: str

    @classmethod
    def load(cls, path: Path) -> QueueModelSettings:
        raw = _read_mapping(path)
        settings = cls(
            holdout_modulus=_as_int(raw.get("holdout_modulus"), "holdout_modulus"),
            holdout_remainder=_as_int(raw.get("holdout_remainder"), "holdout_remainder"),
            train_stride_minutes=_as_int(raw.get("train_stride_minutes"), "train_stride_minutes"),
            eval_separation_minutes=_as_int(raw.get("eval_separation_minutes"), "eval_separation_minutes"),
            short_steps=_as_int(raw.get("short_steps"), "short_steps"),
            quiet_keep=_as_float(raw.get("quiet_keep"), "quiet_keep"),
            max_positive_rows=_as_int(raw.get("max_positive_rows"), "max_positive_rows"),
            max_negative_rows=_as_int(raw.get("max_negative_rows"), "max_negative_rows"),
            min_positive_rows=_as_int(raw.get("min_positive_rows"), "min_positive_rows"),
            min_windows=_as_int(raw.get("min_windows"), "min_windows"),
            min_iou_gain=_as_float(raw.get("min_iou_gain"), "min_iou_gain"),
            decision_thresholds=_as_float_list(raw.get("decision_thresholds"), "decision_thresholds"),
            learning_rate=_as_float(raw.get("learning_rate"), "learning_rate"),
            max_iter=_as_int(raw.get("max_iter"), "max_iter"),
            max_leaf_nodes=_as_int(raw.get("max_leaf_nodes"), "max_leaf_nodes"),
            min_samples_leaf=_as_int(raw.get("min_samples_leaf"), "min_samples_leaf"),
            l2_regularization=_as_float(raw.get("l2_regularization"), "l2_regularization"),
            random_state=_as_int(raw.get("random_state"), "random_state"),
            artifact_dir=str(raw.get("artifact_dir") or ""),
            separate_conditions=_as_bool(raw.get("separate_conditions", False), "separate_conditions"),
            base_artifact=str(raw.get("base_artifact") or ""),
            previous_report=str(raw.get("previous_report") or ""),
        )
        if settings.holdout_modulus <= 1:
            raise ConfigError("holdout_modulus должен быть больше 1")
        if not 0 <= settings.holdout_remainder < settings.holdout_modulus:
            raise ConfigError("holdout_remainder должен лежать в диапазоне modulus")
        if settings.train_stride_minutes <= 0 or settings.eval_separation_minutes <= 0:
            raise ConfigError("шаг обучения и разнос окон должны быть положительными")
        if settings.short_steps <= 0:
            raise ConfigError("short_steps должен быть положительным")
        if not 0 <= settings.quiet_keep <= 1:
            raise ConfigError("quiet_keep должен лежать в [0, 1]")
        if settings.max_positive_rows <= 0 or settings.max_negative_rows <= 0:
            raise ConfigError("размеры выборки должны быть положительными")
        if settings.min_positive_rows <= 0 or settings.min_windows <= 0:
            raise ConfigError("минимумы окон и положительных строк должны быть положительными")
        if settings.min_iou_gain < 0:
            raise ConfigError("min_iou_gain не может быть отрицательным")
        if any(not 0 < item < 1 for item in settings.decision_thresholds):
            raise ConfigError("пороги вероятности должны лежать в (0, 1)")
        if settings.learning_rate <= 0 or settings.max_iter <= 0 or settings.max_leaf_nodes <= 1:
            raise ConfigError("параметры дерева должны быть положительными")
        if settings.min_samples_leaf <= 0 or settings.l2_regularization < 0:
            raise ConfigError("лист и регуляризация заданы неверно")
        if not settings.artifact_dir:
            raise ConfigError("queue_model.artifact_dir должен быть путём")
        return settings

    def holds_out(self, panel: str, date_text: str) -> bool:
        """Тот же остаток, что у профиля State: день проверки модель не видит при выборе порога."""
        digest = hashlib.blake2b(f"{panel}|{date_text}".encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.holdout_modulus == self.holdout_remainder
