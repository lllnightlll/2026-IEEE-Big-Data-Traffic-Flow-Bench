"""Загрузка конфигурации.

Числа оценки, раскладка релиза и путь к пакету живут в разных файлах.
Алгоритм получает уже собранный объект и не открывает YAML сам.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from trafficflowbench.errors import ConfigError


def _read_mapping(path: Path) -> dict:
    """Прочитать YAML-словарь. Пустой файл и список на верхнем уровне — ошибка конфигурации."""
    if not path.is_file():
        raise ConfigError(f"нет файла конфигурации: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} должен быть словарём, получено {type(loaded).__name__}")
    return loaded


def _require(mapping: dict, key: str, source: Path) -> object:
    if key not in mapping:
        raise ConfigError(f"{source} не содержит ключ {key}")
    return mapping[key]


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} должно быть числом, получено {value!r}")
    return float(value)


def _as_int(value: object, label: str) -> int:
    """Целые контракта не принимаются как bool и не усекаются из дроби молча."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} должно быть целым числом, получено {value!r}")
    return value


def _weights_sum_to_one(weights: dict[str, float], label: str) -> None:
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ConfigError(f"веса {label} суммируются в {total}, ожидается 1")


@dataclass(frozen=True)
class ScoringContract:
    """Веса и шкалы официальной оценки. Экземпляр неизменяем: контракт не подкручивается по ходу прогона."""

    total_weights: dict[str, float]
    state_speed_weight: float
    state_flow_weight: float
    speed_rmse_scale_kmh: float
    flow_rmse_scale_vph_per_lane: float
    eligible_pct_observed_min: float
    regimes: dict[str, float]
    queue_free_speed_fraction: float
    queue_history_minutes: int
    queue_horizon_minutes: int
    queue_step_minutes: int
    physics_fd_parts: int
    physics_lwr_parts: int
    physics_low_flow_vph_per_cell: float
    physics_low_flow_fraction_that_zeros_fd: float
    ramp_eligible_pct_observed_min: float
    odme_lambda: float
    odme_weights: dict[str, float]
    published_state_panel: str
    published_state_split: str
    published_state_score: float
    published_state_tolerance: float

    @classmethod
    def load(cls, path: Path) -> ScoringContract:
        raw = _read_mapping(path)
        total = _require(raw, "total_weights", path)
        state = _require(raw, "state", path)
        queue = _require(raw, "queue", path)
        physics = _require(raw, "physics", path)
        odme = _require(raw, "odme", path)
        if not isinstance(total, dict) or not isinstance(state, dict):
            raise ConfigError(f"{path}: total_weights и state должны быть словарями")
        if not isinstance(queue, dict) or not isinstance(physics, dict) or not isinstance(odme, dict):
            raise ConfigError(f"{path}: queue, physics и odme должны быть словарями")

        total_weights = {str(key): _as_float(value, f"total_weights.{key}") for key, value in total.items()}
        _weights_sum_to_one(total_weights, "итога")

        regimes_raw = state.get("regimes")
        if not isinstance(regimes_raw, dict) or not regimes_raw:
            raise ConfigError(f"{path}: state.regimes должен быть непустым словарём")
        regimes = {str(key): _as_float(value, f"state.regimes.{key}") for key, value in regimes_raw.items()}

        odme_weights_raw = odme.get("weights")
        if not isinstance(odme_weights_raw, dict) or not odme_weights_raw:
            raise ConfigError(f"{path}: odme.weights должен быть непустым словарём")
        odme_weights = {str(key): _as_float(value, f"odme.weights.{key}") for key, value in odme_weights_raw.items()}
        _weights_sum_to_one(odme_weights, "ODME")
        fd_parts = _as_int(physics.get("fd_parts"), "physics.fd_parts")
        lwr_parts = _as_int(physics.get("lwr_parts"), "physics.lwr_parts")
        if fd_parts <= 0 or lwr_parts <= 0:
            raise ConfigError("physics.fd_parts и physics.lwr_parts должны быть положительными")

        speed_weight = _as_float(state.get("speed_weight"), "state.speed_weight")
        flow_weight = _as_float(state.get("flow_weight"), "state.flow_weight")
        _weights_sum_to_one({"speed": speed_weight, "flow": flow_weight}, "State")

        published = _require(raw, "published_checks", path)
        if not isinstance(published, dict):
            raise ConfigError(f"{path}: published_checks должен быть словарём")
        published_panel = published.get("state_panel")
        published_split = published.get("state_split")
        if not isinstance(published_panel, str) or not published_panel:
            raise ConfigError(f"{path}: published_checks.state_panel должен быть строкой")
        if not isinstance(published_split, str) or not published_split:
            raise ConfigError(f"{path}: published_checks.state_split должен быть строкой")

        return cls(
            total_weights=total_weights,
            state_speed_weight=speed_weight,
            state_flow_weight=flow_weight,
            speed_rmse_scale_kmh=_as_float(state.get("speed_rmse_scale_kmh"), "state.speed_rmse_scale_kmh"),
            flow_rmse_scale_vph_per_lane=_as_float(
                state.get("flow_rmse_scale_vph_per_lane"),
                "state.flow_rmse_scale_vph_per_lane",
            ),
            eligible_pct_observed_min=_as_float(
                state.get("eligible_pct_observed_min"),
                "state.eligible_pct_observed_min",
            ),
            regimes=regimes,
            queue_free_speed_fraction=_as_float(queue.get("free_speed_fraction"), "queue.free_speed_fraction"),
            queue_history_minutes=_as_int(queue.get("history_minutes"), "queue.history_minutes"),
            queue_horizon_minutes=_as_int(queue.get("horizon_minutes"), "queue.horizon_minutes"),
            queue_step_minutes=_as_int(queue.get("step_minutes"), "queue.step_minutes"),
            physics_fd_parts=fd_parts,
            physics_lwr_parts=lwr_parts,
            physics_low_flow_vph_per_cell=_as_float(
                physics.get("low_flow_vph_per_cell"),
                "physics.low_flow_vph_per_cell",
            ),
            physics_low_flow_fraction_that_zeros_fd=_as_float(
                physics.get("low_flow_fraction_that_zeros_fd"),
                "physics.low_flow_fraction_that_zeros_fd",
            ),
            ramp_eligible_pct_observed_min=_as_float(
                physics.get("ramp_eligible_pct_observed_min"),
                "physics.ramp_eligible_pct_observed_min",
            ),
            odme_lambda=_as_float(odme.get("lambda"), "odme.lambda"),
            odme_weights=odme_weights,
            published_state_panel=published_panel,
            published_state_split=published_split,
            published_state_score=_as_float(published.get("state_score"), "published_checks.state_score"),
            published_state_tolerance=_as_float(
                published.get("state_tolerance"),
                "published_checks.state_tolerance",
            ),
        )


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"{label} должен быть непустым списком строк")
    return tuple(value)


def _load_merge(raw: dict, path: Path) -> MergeContract:
    """Собрать контракт склейки. Имена задач приходят из YAML, код их не перечисляет заново."""
    merge = _require(raw, "merge", path)
    if not isinstance(merge, dict):
        raise ConfigError(f"{path}: merge должен быть словарём")
    tasks_raw = merge.get("tasks")
    if not isinstance(tasks_raw, dict) or not tasks_raw:
        raise ConfigError(f"{path}: merge.tasks должен быть непустым словарём")
    tasks: list[MergeTask] = []
    for name, spec in tasks_raw.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"{path}: merge.tasks.{name} должен быть словарём")
        tasks.append(
            MergeTask(
                name=str(name),
                keys=_string_tuple(spec.get("keys"), f"{path}: merge.tasks.{name}.keys"),
                values=_string_tuple(spec.get("values"), f"{path}: merge.tasks.{name}.values"),
            )
        )
    return MergeContract(
        chunk_rows=_as_int(merge.get("chunk_rows"), f"{path}: merge.chunk_rows"),
        output_columns=_string_tuple(merge.get("output_columns"), f"{path}: merge.output_columns"),
        tasks=tuple(tasks),
    )


@dataclass(frozen=True)
class MergeTask:
    """Ключ и значения одного блока сдачи. Имена совпадают с колонками Kaggle."""

    name: str
    keys: tuple[str, ...]
    values: tuple[str, ...]


@dataclass(frozen=True)
class MergeContract:
    """Как три файла садятся на submission_key.csv. Порядок tasks — порядок задач, не колонок."""

    chunk_rows: int
    output_columns: tuple[str, ...]
    tasks: tuple[MergeTask, ...]

    def task(self, name: str) -> MergeTask:
        for item in self.tasks:
            if item.name == name:
                return item
        known = ", ".join(item.name for item in self.tasks)
        raise ConfigError(f"в контракте склейки нет задачи {name}. Есть: {known}")


@dataclass(frozen=True)
class ReleaseLayout:
    """Относительные пути внутри корня релиза и обязательные колонки трёх шаблонов."""

    corridors_catalog: str
    unmasked_dir: str
    masked_dir: str
    state_template: str
    queue_template: str
    odme_template: str
    observation_glob: str
    submission_key: str
    state_columns: tuple[str, ...]
    queue_columns: tuple[str, ...]
    odme_columns: tuple[str, ...]
    merge: MergeContract

    @classmethod
    def load(cls, path: Path) -> ReleaseLayout:
        raw = _read_mapping(path)
        paths = _require(raw, "paths", path)
        schemas = _require(raw, "schemas", path)
        if not isinstance(paths, dict) or not isinstance(schemas, dict):
            raise ConfigError(f"{path}: paths и schemas должны быть словарями")

        def text(key: str) -> str:
            value = paths.get(key)
            if not isinstance(value, str) or not value:
                raise ConfigError(f"{path}: paths.{key} должен быть непустой строкой")
            return value

        def columns(key: str) -> tuple[str, ...]:
            value = schemas.get(key)
            if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
                raise ConfigError(f"{path}: schemas.{key} должен быть списком строк")
            return tuple(value)

        return cls(
            corridors_catalog=text("corridors_catalog"),
            unmasked_dir=text("unmasked_dir"),
            masked_dir=text("masked_dir"),
            state_template=text("state_template"),
            queue_template=text("queue_template"),
            odme_template=text("odme_template"),
            observation_glob=text("observation_glob"),
            submission_key=text("submission_key"),
            state_columns=columns("state_columns"),
            queue_columns=columns("queue_columns"),
            odme_columns=columns("odme_columns"),
            merge=_load_merge(raw, path),
        )


@dataclass(frozen=True)
class ReleaseRequest:
    """Какую панель и какой сплит читать. Корень может быть пустым, пока пакет не скачан."""

    release_root: Path | None
    panel: str
    split: str
    submission_splits: tuple[str, ...]
    artifact_dir: str

    @classmethod
    def load(cls, path: Path, release_root_override: Path | None) -> ReleaseRequest:
        raw = _read_mapping(path)
        panel = raw.get("panel")
        split = raw.get("split")
        if not isinstance(panel, str) or not panel:
            raise ConfigError(f"{path}: panel должен быть непустой строкой")
        if not isinstance(split, str) or not split:
            raise ConfigError(f"{path}: split должен быть непустой строкой")
        artifact_dir = raw.get("artifact_dir") or "experiments/artifacts/v1"
        if not isinstance(artifact_dir, str) or not artifact_dir:
            raise ConfigError(f"{path}: artifact_dir должен быть путём относительно корня репозитория")
        # Ключ Kaggle содержит и validation, и private. Одного сплита недостаточно для склейки.
        configured_splits = raw.get("submission_splits")
        if configured_splits is None:
            submission_splits = ("validation", "private")
        elif isinstance(configured_splits, list) and configured_splits and all(
            isinstance(item, str) and item for item in configured_splits
        ):
            submission_splits = tuple(configured_splits)
        else:
            raise ConfigError(f"{path}: submission_splits должен быть непустым списком строк")

        # Приоритет: явный аргумент команды, затем переменная окружения, затем файл.
        # Так скрипт не зашивает каталог машины, а локальный файл не обязан попадать в git.
        if release_root_override is not None:
            root: Path | None = release_root_override
        else:
            from_env = os.environ.get("TFB_RELEASE_ROOT")
            if from_env:
                root = Path(from_env)
            else:
                configured = raw.get("release_root")
                if configured is None:
                    root = None
                elif isinstance(configured, str) and configured.strip():
                    root = Path(configured)
                else:
                    raise ConfigError(f"{path}: release_root должен быть путём или null")
        return cls(
            release_root=root,
            panel=panel,
            split=split,
            submission_splits=submission_splits,
            artifact_dir=artifact_dir,
        )


@dataclass(frozen=True)
class UpstreamCheckout:
    """Локальный клон репозитория организаторов и относительные пути их скриптов."""

    repository: str
    ref: str
    commit: str
    checkout: Path
    scripts: dict[str, Path]

    @classmethod
    def load(cls, repo_root: Path, path: Path) -> UpstreamCheckout:
        raw = _read_mapping(path)
        checkout_raw = raw.get("checkout")
        scripts_raw = raw.get("scripts")
        repository = raw.get("repository")
        ref = raw.get("ref")
        commit = raw.get("commit")
        if not isinstance(checkout_raw, str) or not checkout_raw:
            raise ConfigError(f"{path}: checkout должен быть путём относительно корня репозитория")
        if not isinstance(scripts_raw, dict) or not scripts_raw:
            raise ConfigError(f"{path}: scripts должен быть непустым словарём")
        if not all(isinstance(value, str) and value for value in (repository, ref, commit)):
            raise ConfigError(f"{path}: repository, ref и commit должны быть строками")
        checkout = (repo_root / checkout_raw).resolve()
        scripts: dict[str, Path] = {}
        for name, relative in scripts_raw.items():
            if not isinstance(relative, str) or not relative:
                raise ConfigError(f"{path}: scripts.{name} должен быть путём")
            scripts[str(name)] = checkout / relative
        loaded = cls(
            repository=str(repository),
            ref=str(ref),
            commit=str(commit),
            checkout=checkout,
            scripts=scripts,
        )
        loaded.require_present()
        return loaded

    def require_present(self) -> None:
        """Клон должен лежать там, куда указывает checkout. Иначе оценки писать нечем."""
        if not self.checkout.is_dir():
            raise ConfigError(
                f"нет клона организаторов: {self.checkout}. "
                f"Ожидается {self.repository} на коммите {self.commit}."
            )
        missing = [f"{name}: {path}" for name, path in self.scripts.items() if not path.is_file()]
        if missing:
            raise ConfigError("в клоне организаторов нет скриптов:\n" + "\n".join(missing))

    def script(self, name: str) -> Path:
        try:
            return self.scripts[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.scripts))
            raise ConfigError(f"скрипт {name} не описан в upstream.yaml. Есть: {known}") from exc


@dataclass(frozen=True)
class ProjectConfig:
    """Всё, что каркас читает до первого обращения к данным."""

    repo_root: Path
    scoring: ScoringContract
    layout: ReleaseLayout
    request: ReleaseRequest
    upstream: UpstreamCheckout

    @classmethod
    def load(
        cls,
        repo_root: Path,
        release_config: Path | None = None,
        release_root_override: Path | None = None,
    ) -> ProjectConfig:
        config_dir = repo_root / "config"
        chosen_release = release_config
        if chosen_release is None:
            local = config_dir / "release.local.yaml"
            chosen_release = local if local.is_file() else config_dir / "release.example.yaml"
        return cls(
            repo_root=repo_root,
            scoring=ScoringContract.load(config_dir / "scoring.yaml"),
            layout=ReleaseLayout.load(config_dir / "layout.yaml"),
            request=ReleaseRequest.load(chosen_release, release_root_override),
            upstream=UpstreamCheckout.load(repo_root, config_dir / "upstream.yaml"),
        )
