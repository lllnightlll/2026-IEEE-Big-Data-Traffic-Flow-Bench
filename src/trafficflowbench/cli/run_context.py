"""Команда слоя v2: профиль и контекст на одной панели, сверка официальным оценщиком."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from trafficflowbench.cli.read_panel import _configure_stdio, repo_root_from
from trafficflowbench.config_loader import ProjectConfig
from trafficflowbench.data.reader import ReleaseRoot
from trafficflowbench.errors import ProjectError, ReleaseNotFound
from trafficflowbench.external.organizer import OrganizerRunner, ScoreSheet, current_python
from trafficflowbench.state.context_model import PanelContextTrainer, StateSubmissionWriter
from trafficflowbench.state.settings import ContextSettings

OBSERVATION_COLUMNS = [
    "timestamp",
    "station_id",
    "link_id",
    "speed_kmh",
    "flow_vph",
    "pct_observed",
]


class RunContextCommand:
    """Обучить контекст на train и записать сдачу выбранного сплита."""

    def __init__(self, argv: list[str] | None = None) -> None:
        self.argv = argv if argv is not None else sys.argv[1:]

    def run(self) -> int:
        parser = argparse.ArgumentParser(description="Восстановить State профилем и соседними наблюдениями")
        parser.add_argument("--release-root", type=Path, default=None)
        parser.add_argument("--panel", default=None)
        parser.add_argument("--split", default=None, help="сплит сдачи, обычно validation")
        parser.add_argument("--all-panels", action="store_true", help="все панели и оба сплита ключа Kaggle")
        parser.add_argument("--score-train", action="store_true", help="сравнить с профилем на train официальным оценщиком")
        args = parser.parse_args(self.argv)
        if args.all_panels and (args.panel or args.split or args.score_train):
            raise ProjectError("полный прогон не совмещается с --panel, --split и --score-train")

        root = repo_root_from(Path(__file__))
        config = ProjectConfig.load(root, release_root_override=args.release_root)
        settings = ContextSettings.load(root / "config" / "context.yaml")
        if config.request.release_root is None:
            raise ReleaseNotFound("путь к релизу не задан")
        release = ReleaseRoot(config.request.release_root)
        from trafficflowbench.data.catalog import CorridorCatalog

        catalog = CorridorCatalog.load(release.path / config.layout.corridors_catalog)
        if args.all_panels:
            panels = list(catalog.ids())
            splits = list(config.request.submission_splits)
        else:
            panel = args.panel or config.request.panel
            catalog.require(panel)
            panels = [panel]
            splits = [args.split or config.request.split]
        minutes = config.scoring.queue_step_minutes
        lines: list[str] = []
        for panel in panels:
            panel_dir = release.path / "corridors" / panel
            trainer = PanelContextTrainer(panel, panel_dir, config.layout, settings)
            artifact_dir = root / settings.artifact_dir / panel
            artifact_dir.mkdir(parents=True, exist_ok=True)
            if args.score_train:
                lines.extend(self._score_train(trainer, config, settings, artifact_dir, minutes, release.path))
            print(f"[context] {panel}", flush=True)
            model = trainer.fit(set(), minutes)
            for split in splits:
                submission = artifact_dir / f"state_{split}.csv"
                written = self._write_split(model, trainer, config.layout, split, submission)
                print(f"  {split}: {written:,} строк", flush=True)
                lines.append(f"{panel} {split}: {written}")
        if args.all_panels:
            upload, written = self._assemble(config, settings, panels, splits, release.path)
            lines.append(f"submission: {upload}")
            lines.append(f"submission_rows: {written}")
        sys.stdout.write("\n".join(lines) + "\n")
        return 0

    def _score_train(
        self,
        trainer: PanelContextTrainer,
        config: ProjectConfig,
        settings: ContextSettings,
        artifact_dir: Path,
        minutes: int,
        release_root: Path,
    ) -> list[str]:
        held = trainer.holdout_dates()
        if not held:
            raise ProjectError(f"отложенных дней не нашлось у {trainer.panel}")
        model = trainer.fit(held, minutes)
        template = release_root / "task1" / trainer.panel / "train" / "sample_submission_state.csv"
        context_path = artifact_dir / "state_train_context.csv"
        profile_path = artifact_dir / "state_train_profile.csv"
        self._write_split(model, trainer, config.layout, "train", context_path, profile_only=False)
        self._write_split(model, trainer, config.layout, "train", profile_path, profile_only=True)
        runner = OrganizerRunner(current_python(), config.upstream)
        context_score = artifact_dir / "state_train_context_score.csv"
        profile_score = artifact_dir / "state_train_profile_score.csv"
        for submission, score in ((context_path, context_score), (profile_path, profile_score)):
            runner.run(
                "state_score",
                [
                    "--submission", str(submission),
                    "--release-root", str(release_root),
                    "--split", "train",
                    "--output", str(score),
                    "--panel", trainer.panel,
                ],
            )
        context_sheet = ScoreSheet(context_score)
        profile_sheet = ScoreSheet(profile_score)
        lines = [f"holdout_days: {len(held)}"]
        for regime in config.scoring.regimes:
            context_value = _regime_score(context_sheet, regime)
            profile_value = _regime_score(profile_sheet, regime)
            lines.append(f"S_state {regime}: context {context_value:.6f} profile {profile_value:.6f}")
        lines.append(
            f"S_state train: context {context_sheet.value('panel', 'S_state', trainer.panel):.6f} "
            f"profile {profile_sheet.value('panel', 'S_state', trainer.panel):.6f}"
        )
        return lines

    @staticmethod
    def _write_split(
        model,
        trainer: PanelContextTrainer,
        layout: ReleaseLayout,
        split: str,
        output: Path,
        profile_only: bool = False,
    ) -> int:
        pieces: list[pd.DataFrame] = []
        for path in trainer.masked_files(split):
            frame = pd.read_parquet(path, columns=OBSERVATION_COLUMNS)
            if profile_only:
                frame = frame.copy()
                frame["speed_kmh"] = pd.NA
                frame["flow_vph"] = pd.NA
            pieces.append(model.predict_observations(frame))
        if not pieces:
            raise ProjectError(f"нет маскированных наблюдений {trainer.panel}/{split}")
        predictions = pd.concat(pieces, ignore_index=True)
        template = trainer.panel_dir.parent.parent / "task1" / trainer.panel / split / "sample_submission_state.csv"
        return StateSubmissionWriter(layout.state_columns).write(template, predictions, output)

    @staticmethod
    def _assemble(
        config: ProjectConfig,
        settings: ContextSettings,
        panels: list[str],
        splits: list[str],
        release_root: Path,
    ) -> tuple[Path, int]:
        """State берётся из v2, очередь и ODME остаются файлами принятой базы v1."""
        from trafficflowbench.baseline.pipeline import _concat_csv
        from trafficflowbench.submission.assembler import SubmissionAssembler

        version_dir = config.repo_root / settings.artifact_dir
        baseline_dir = config.repo_root / config.request.artifact_dir
        state_parts: list[Path] = []
        queue_parts: list[Path] = []
        odme_parts: list[Path] = []
        for split in splits:
            split_state = version_dir / f"state_{split}.csv"
            _concat_csv([version_dir / panel / f"state_{split}.csv" for panel in panels], split_state)
            state_parts.append(split_state)
            queue_parts.append(baseline_dir / f"queue_{split}.csv")
            odme_parts.append(baseline_dir / f"odme_{split}" / "baseline_submission.csv")
        state_file = version_dir / "state_submission.csv"
        queue_file = version_dir / "queue_submission.csv"
        odme_file = version_dir / "odme_submission.csv"
        _concat_csv(state_parts, state_file)
        _concat_csv(queue_parts, queue_file)
        _concat_csv(odme_parts, odme_file)
        upload = version_dir / "submission.csv"
        written = SubmissionAssembler(config.layout.merge).assemble(
            {"state": state_file, "queue": queue_file, "odme": odme_file},
            release_root / config.layout.submission_key,
            upload,
        )
        return upload, written


def _regime_score(sheet: ScoreSheet, regime: str) -> float:
    rows = sheet.frame[(sheet.frame["level"] == "panel_regime") & (sheet.frame["regime"] == regime)]
    if rows.empty:
        raise ProjectError(f"в оценке нет режима {regime}")
    return float(rows.iloc[0]["S_state"])


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    try:
        return RunContextCommand(argv).run()
    except ProjectError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
