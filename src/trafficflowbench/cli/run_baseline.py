"""Команда базы v1. Панели берутся из каталога релиза или из флага, не из списка в коде."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trafficflowbench.baseline.pipeline import BaselinePipeline
from trafficflowbench.cli.read_panel import _configure_stdio, repo_root_from
from trafficflowbench.config_loader import ProjectConfig
from trafficflowbench.data.catalog import CorridorCatalog
from trafficflowbench.data.reader import ReleaseRoot
from trafficflowbench.errors import ProjectError, ReleaseNotFound
from trafficflowbench.external.organizer import current_python


class RunBaselineCommand:
    """Собрать аргументы и отдать их конвейеру базы."""

    def __init__(self, argv: list[str] | None = None) -> None:
        self.argv = argv if argv is not None else sys.argv[1:]

    def run(self) -> int:
        parser = argparse.ArgumentParser(description="Прогнать базовые решения организаторов")
        parser.add_argument("--release-root", type=Path, default=None, help="каталог kaggle_public")
        parser.add_argument("--release-config", type=Path, default=None)
        parser.add_argument("--panel", action="append", default=None, help="одна или несколько панелей")
        parser.add_argument("--all-panels", action="store_true", help="все панели из corridors.json релиза")
        args = parser.parse_args(self.argv)
        if args.all_panels and args.panel:
            raise ProjectError("укажите либо --all-panels, либо --panel, но не оба сразу")

        root = repo_root_from(Path(__file__))
        config = ProjectConfig.load(root, args.release_config, args.release_root)
        if config.request.release_root is None:
            raise ReleaseNotFound(
                "путь к релизу не задан. Запишите каталог kaggle_public в config/release.local.yaml "
                "или передайте --release-root."
            )
        if args.all_panels:
            release = ReleaseRoot(config.request.release_root)
            panels = list(
                CorridorCatalog.load(release.path / config.layout.corridors_catalog).ids()
            )
        else:
            panels = list(args.panel) if args.panel else [config.request.panel]

        sys.stdout.write(BaselinePipeline(config, current_python()).run(panels))
        return 0


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    try:
        return RunBaselineCommand(argv).run()
    except ProjectError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
