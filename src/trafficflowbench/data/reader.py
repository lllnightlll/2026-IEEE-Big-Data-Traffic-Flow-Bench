"""Чтение одной панели релиза.

Фаза A специально ограничена учётом шаблонов и слоёв наблюдений.
Немаскированный слой есть только у train. Его отсутствие на validation и private
не ошибка: это устройство пакета, а не сбой распаковки.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trafficflowbench.config_loader import ReleaseLayout
from trafficflowbench.data.catalog import Corridor, CorridorCatalog
from trafficflowbench.errors import ReleaseNotFound, TemplateNotFound, TemplateSchemaError


@dataclass(frozen=True)
class LayerCensus:
    """Есть ли каталог слоя и сколько в нём parquet-файлов. Ноль файлов — законный ответ."""

    directory: Path
    present: bool
    file_count: int


@dataclass(frozen=True)
class TemplateCensus:
    """Число строк данных в шаблоне сдачи. rows is None, если файл задачи не обязателен и его нет."""

    path: Path
    present: bool
    rows: int | None


@dataclass(frozen=True)
class PanelCensus:
    """Сводка, которую печатает команда чтения. Моделей здесь ещё нет."""

    corridor: Corridor
    split: str
    state_template: TemplateCensus
    queue_template: TemplateCensus
    odme_template: TemplateCensus
    unmasked: LayerCensus
    masked: LayerCensus


class ReleaseRoot:
    """Проверенный корень kaggle_public. Дальше пути считаются только от него."""

    def __init__(self, path: Path) -> None:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise ReleaseNotFound(
                f"корень релиза не существует: {resolved}\n"
                "Скачайте пакет со страницы Data, распакуйте его и укажите каталог kaggle_public "
                "в config/release.local.yaml, в TFB_RELEASE_ROOT или флагом --release-root. "
                "Нужна пересборка после 11 сентября 2026. Указывать надо сам kaggle_public, не родительский каталог."
            )
        if not resolved.is_dir():
            raise ReleaseNotFound(f"корень релиза не каталог: {resolved}")
        self.path = resolved

    def locate(self, template: str, panel: str, split: str) -> Path:
        """Подставить панель и сплит в шаблон пути из config/layout.yaml."""
        try:
            relative = template.format(panel=panel, split=split)
        except KeyError as exc:
            raise ReleaseNotFound(
                f"шаблон пути {template!r} требует поле {exc.args[0]}, а читатель передаёт только panel и split"
            ) from exc
        return self.path / relative


class CsvTemplateCounter:
    """Считает строки шаблона потоком, не загружая месяц коридора в память.

    Заодно сверяет заголовок со схемой. Повторная сериализация значений здесь не делается:
    метки времени должны остаться байтами релиза, а подсчёт строк их не переписывает.
    """

    def __init__(self, expected_columns: tuple[str, ...]) -> None:
        self.expected_columns = expected_columns

    def count(self, path: Path) -> TemplateCensus:
        if not path.is_file():
            return TemplateCensus(path=path, present=False, rows=None)
        with path.open(encoding="utf-8", newline="") as handle:
            header = handle.readline()
            if not header:
                raise TemplateSchemaError(f"пустой шаблон: {path}")
            # BOM встречается у файлов, сохранённых из табличного редактора, и ломает имя первой колонки.
            columns = [column.strip().lstrip("\ufeff") for column in header.strip().split(",")]
            missing = [column for column in self.expected_columns if column not in columns]
            if missing:
                raise TemplateSchemaError(
                    f"{path} не содержит колонки {missing}. Заголовок: {', '.join(columns)}"
                )
            rows = sum(1 for line in handle if line.strip())
        return TemplateCensus(path=path, present=True, rows=rows)


class ObservationLayer:
    """Слой parquet одного сплита. Каталог может отсутствовать: тогда present=False и список пуст."""

    def __init__(self, directory: Path, observation_glob: str) -> None:
        self.directory = directory
        self.observation_glob = observation_glob

    def census(self) -> LayerCensus:
        if not self.directory.is_dir():
            return LayerCensus(directory=self.directory, present=False, file_count=0)
        files = sorted(self.directory.glob(self.observation_glob))
        return LayerCensus(
            directory=self.directory,
            present=True,
            file_count=sum(1 for path in files if path.is_file()),
        )


class PanelReader:
    """Читает одну панель: каталог коридоров, три шаблона и оба слоя наблюдений."""

    def __init__(self, root: ReleaseRoot, layout: ReleaseLayout) -> None:
        self.root = root
        self.layout = layout
        catalog_path = root.path / layout.corridors_catalog
        self.catalog = CorridorCatalog.load(catalog_path)

    def read(self, panel: str, split: str) -> PanelCensus:
        corridor = self.catalog.require(panel)
        state_path = self.root.locate(self.layout.state_template, panel, split)
        state = CsvTemplateCounter(self.layout.state_columns).count(state_path)
        if not state.present or state.rows is None:
            raise TemplateNotFound(
                f"нет шаблона State для {panel}/{split}: {state_path}\n"
                "Цели Task 1 берутся из этого файла, а не из пустых ячеек наблюдений."
            )
        return PanelCensus(
            corridor=corridor,
            split=split,
            state_template=state,
            queue_template=CsvTemplateCounter(self.layout.queue_columns).count(
                self.root.locate(self.layout.queue_template, panel, split)
            ),
            odme_template=CsvTemplateCounter(self.layout.odme_columns).count(
                self.root.locate(self.layout.odme_template, panel, split)
            ),
            unmasked=ObservationLayer(
                self.root.locate(self.layout.unmasked_dir, panel, split),
                self.layout.observation_glob,
            ).census(),
            masked=ObservationLayer(
                self.root.locate(self.layout.masked_dir, panel, split),
                self.layout.observation_glob,
            ).census(),
        )
