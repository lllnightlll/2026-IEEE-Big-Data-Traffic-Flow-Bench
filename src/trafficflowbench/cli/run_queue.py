"""Команда очереди: ongoing со сдвигом, onset по повторяемости train, склейка полной сдачи."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from trafficflowbench.cli.read_panel import _configure_stdio, repo_root_from
from trafficflowbench.config_loader import ProjectConfig
from trafficflowbench.data.reader import ReleaseRoot
from trafficflowbench.errors import ProjectError, ReleaseNotFound
from trafficflowbench.queue.builder import QueuePanelBuilder
from trafficflowbench.queue.settings import QueueSettings
from trafficflowbench.state.settings import ContextSettings


class RunQueueCommand:
    """Собрать очередь на все коридоры Task 2 и посадить её на State v2 и ODME базы."""

    def __init__(self, argv: list[str] | None = None) -> None:
        self.argv = argv if argv is not None else sys.argv[1:]

    def run(self) -> int:
        parser = argparse.ArgumentParser(description="Собрать сдачу очереди и полный CSV")
        parser.add_argument("--release-root", type=Path, default=None)
        args = parser.parse_args(self.argv)

        root = repo_root_from(Path(__file__))
        config = ProjectConfig.load(root, release_root_override=args.release_root)
        settings = QueueSettings.load(root / "config" / "queue.yaml")
        context = ContextSettings.load(root / "config" / "context.yaml")
        if config.request.release_root is None:
            raise ReleaseNotFound("путь к релизу не задан")
        release = ReleaseRoot(config.request.release_root)
        panels = self._panels(release.path, config.request.submission_splits)
        if not panels:
            raise ProjectError("в релизе нет шаблонов Task 2")
        version_dir = root / settings.artifact_dir
        version_dir.mkdir(parents=True, exist_ok=True)
        by_split: dict[str, list[pd.DataFrame]] = {split: [] for split in config.request.submission_splits}
        for panel in panels:
            print(f"[queue] {panel}", flush=True)
            builder = QueuePanelBuilder(
                panel,
                release.path,
                settings,
                config.scoring.queue_free_speed_fraction,
                config.scoring.queue_step_minutes,
            )
            for split in config.request.submission_splits:
                frame = builder.predict_split(split)
                self._report(split, frame)
                by_split[split].append(frame)
        queue_file = version_dir / "queue_submission.csv"
        queue_frame = pd.concat(
            [piece for split in config.request.submission_splits for piece in by_split[split]],
            ignore_index=True,
        )
        queue_frame.to_csv(queue_file, index=False)
        upload, written = self._assemble(config, context, settings, release.path, queue_file)
        print(f"submission: {upload}", flush=True)
        print(f"submission_rows: {written}", flush=True)
        print(f"queue_rows: {len(queue_frame)}", flush=True)
        print(f"queue_ones: {int(queue_frame['queue_pred'].sum())}", flush=True)
        return 0

    @staticmethod
    def _panels(release_root: Path, splits: tuple[str, ...]) -> list[str]:
        task_root = release_root / "task2"
        if not task_root.is_dir():
            raise ProjectError(f"нет каталога Task 2: {task_root}")
        found: list[str] = []
        for panel_dir in sorted(path for path in task_root.iterdir() if path.is_dir()):
            if any((panel_dir / split / "sample_submission_queue.csv").is_file() for split in splits):
                found.append(panel_dir.name)
        return found

    @staticmethod
    def _report(split: str, frame: pd.DataFrame) -> None:
        ones = int(frame["queue_pred"].sum())
        print(f"  {split}: {len(frame):,} строк, единицы {ones:,}", flush=True)

    @staticmethod
    def _assemble(config: ProjectConfig, context: ContextSettings, settings: QueueSettings, release_root: Path, queue_file: Path) -> tuple[Path, int]:
        """State и ODME остаются файлами уже принятой сдачи v2. Очередь заменяется целиком."""
        from trafficflowbench.submission.assembler import SubmissionAssembler

        previous = config.repo_root / context.artifact_dir
        state_file = previous / "state_submission.csv"
        odme_file = previous / "odme_submission.csv"
        version_dir = config.repo_root / settings.artifact_dir
        upload = version_dir / "submission.csv"
        written = SubmissionAssembler(config.layout.merge).assemble(
            {"state": state_file, "queue": queue_file, "odme": odme_file},
            release_root / config.layout.submission_key,
            upload,
        )
        return upload, written


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    try:
        return RunQueueCommand(argv).run()
    except ProjectError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
