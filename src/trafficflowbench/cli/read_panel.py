"""Команда чтения панели.

Печатает число строк шаблона State и отмечает, есть ли немаскированный слой.
На validation и private этого слоя нет, и команда из-за этого не останавливается.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trafficflowbench.config_loader import ProjectConfig
from trafficflowbench.data.reader import LayerCensus, PanelCensus, PanelReader, ReleaseRoot, TemplateCensus
from trafficflowbench.errors import ProjectError


def repo_root_from(script_file: Path) -> Path:
    """Корень репозитория — родитель каталога scripts либо src, в зависимости от точки входа."""
    start = script_file.resolve().parent
    for candidate in (start, *start.parents):
        if (candidate / "config" / "layout.yaml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise ProjectError(f"не найден корень репозитория от {script_file}")


class ReadPanelCommand:
    """Собрать конфигурацию, прочитать одну панель и напечатать сводку."""

    def __init__(self, argv: list[str] | None = None) -> None:
        self.argv = argv if argv is not None else sys.argv[1:]

    def run(self) -> int:
        parser = argparse.ArgumentParser(description="Прочитать одну панель релиза TrafficFlowBench")
        parser.add_argument("--release-root", type=Path, default=None, help="каталог kaggle_public")
        parser.add_argument("--release-config", type=Path, default=None, help="YAML с путём, панелью и сплитом")
        parser.add_argument("--panel", default=None, help="идентификатор панели из corridors.json")
        parser.add_argument("--split", default=None, help="train, validation или private")
        args = parser.parse_args(self.argv)

        root = repo_root_from(Path(__file__))
        config = ProjectConfig.load(root, args.release_config, args.release_root)
        panel = args.panel or config.request.panel
        split = args.split or config.request.split
        if config.request.release_root is None:
            raise ProjectError(
                "путь к релизу не задан. Скопируйте config/release.example.yaml в config/release.local.yaml "
                "и запишите туда каталог kaggle_public, либо передайте --release-root или TFB_RELEASE_ROOT."
            )

        census = PanelReader(ReleaseRoot(config.request.release_root), config.layout).read(panel, split)
        sys.stdout.write(self.format(census))
        return 0

    def format(self, census: PanelCensus) -> str:
        lines = [
            f"panel: {census.corridor.corridor_id}",
            f"family: {census.corridor.family_id}",
            f"split: {census.split}",
            f"state_template_rows: {census.state_template.rows}",
            f"state_template: {census.state_template.path}",
            self._template_line("queue", census.queue_template),
            self._template_line("odme", census.odme_template),
            self._layer_line("unmasked", census.unmasked),
            self._layer_line("masked", census.masked),
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _template_line(name: str, template: TemplateCensus) -> str:
        if template.present:
            return f"{name}_template_rows: {template.rows}"
        return f"{name}_template_rows: absent"

    @staticmethod
    def _layer_line(name: str, layer: LayerCensus) -> str:
        # Отсутствие каталога и пустой каталог различаются: первое ожидается на validation,
        # второе может означать оборванную распаковку.
        state = "present" if layer.present else "absent"
        return f"{name}_layer: {state} files={layer.file_count}"


def _configure_stdio() -> None:
    """На Windows консоль часто не UTF-8, а сообщения каркаса написаны по-русски."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            continue


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    try:
        return ReadPanelCommand(argv).run()
    except ProjectError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
