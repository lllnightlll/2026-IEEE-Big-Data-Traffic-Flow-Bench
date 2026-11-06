"""Команда поправки State. Панель заменяет контекст только после прироста на отложенных днях."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from trafficflowbench.cli.read_panel import _configure_stdio, repo_root_from
from trafficflowbench.config_loader import ProjectConfig
from trafficflowbench.data.reader import ReleaseRoot
from trafficflowbench.errors import ProjectError, ReleaseNotFound
from trafficflowbench.state.context_model import PanelContextTrainer, StateSubmissionWriter
from trafficflowbench.state.residual import MASKED_COLUMNS, ResidualPanelStudy, StateResidualSettings
from trafficflowbench.state.settings import ContextSettings


class RunStateResidualCommand:
    """Сравнить поправку с контекстом и собрать сдачу, если отложенные дни это подтверждают."""

    def __init__(self, argv: list[str] | None = None) -> None:
        self.argv = argv if argv is not None else sys.argv[1:]

    def run(self) -> int:
        parser = argparse.ArgumentParser(description="Поправить State деревом поверх контекста")
        parser.add_argument("--release-root", type=Path, default=None)
        parser.add_argument("--panel", default=None)
        parser.add_argument("--evaluate-only", action="store_true")
        args = parser.parse_args(self.argv)
        root = repo_root_from(Path(__file__))
        config = ProjectConfig.load(root, release_root_override=args.release_root)
        if config.request.release_root is None:
            raise ReleaseNotFound("путь к релизу не задан")
        release = ReleaseRoot(config.request.release_root)
        context_settings = ContextSettings.load(root / "config" / "context.yaml")
        residual_settings = StateResidualSettings.load(root / "config" / "state_residual.yaml")
        from trafficflowbench.data.catalog import CorridorCatalog

        catalog = CorridorCatalog.load(release.path / config.layout.corridors_catalog)
        panels = [args.panel] if args.panel else list(catalog.ids())
        if args.panel:
            catalog.require(args.panel)
        version_dir = root / residual_settings.artifact_dir
        version_dir.mkdir(parents=True, exist_ok=True)
        report = version_dir / "holdout_state.csv"
        if report.exists():
            report.unlink()
        adopted: list[str] = []
        for panel in panels:
            print(f"[state-residual] {panel}", flush=True)
            if self._panel(root, release.path, config, context_settings, residual_settings, panel, args.evaluate_only):
                adopted.append(panel)
        report_path = version_dir / "holdout_state.csv"
        print(f"holdout: {report_path}", flush=True)
        if args.evaluate_only or not adopted:
            print("submission: контекст v2 остаётся, новая сдача не собрана", flush=True)
            return 0
        queue_file = _queue_file(root, residual_settings)
        upload, written = _assemble(root, config, context_settings, residual_settings, release.path, panels, queue_file)
        print(f"submission: {upload}", flush=True)
        print(f"submission_rows: {written}", flush=True)
        print(f"state_panels: {', '.join(adopted)}", flush=True)
        return 0

    def _panel(
        self,
        root: Path,
        release_root: Path,
        config: ProjectConfig,
        context_settings: ContextSettings,
        residual_settings: StateResidualSettings,
        panel: str,
        evaluate_only: bool,
    ) -> bool:
        panel_dir = release_root / "corridors" / panel
        trainer = PanelContextTrainer(panel, panel_dir, config.layout, context_settings)
        held = trainer.holdout_dates()
        if not held:
            raise ProjectError(f"отложенных дней не нашлось у {panel}")
        minutes = config.scoring.queue_step_minutes
        model = trainer.fit(held, minutes)
        from trafficflowbench.state.corridor import DiagramBounds, LinkChain

        chain = LinkChain.read(panel_dir)
        diagram = DiagramBounds.read(panel_dir, context_settings.speed_epsilon_kmh)
        study = ResidualPanelStudy(panel_dir, context_settings, residual_settings, config.scoring, chain, diagram)
        print("  обучение поправки", flush=True)
        corrector = study.train(model, held)
        if corrector is None:
            _append_report(root, residual_settings, panel, "", "", "skip")
            print("  мало вырезанных ячеек, панель остаётся контекстом", flush=True)
            return False
        print("  счёт на отложенных днях", flush=True)
        context_value, corrected_value = study.score(model, corrector, held)
        gain = corrected_value - context_value
        choice = "residual" if gain >= residual_settings.min_state_gain else "context"
        _append_report(root, residual_settings, panel, f"{context_value:.6f}", f"{corrected_value:.6f}", choice)
        print(f"  context {context_value:.6f} residual {corrected_value:.6f} {choice}", flush=True)
        if choice != "residual" or evaluate_only:
            return False
        print("  запись validation и private", flush=True)
        full_model = trainer.fit(set(), minutes)
        full_corrector = study.train(full_model, set())
        if full_corrector is None:
            raise ProjectError(f"поправка {panel} не собралась на полном train")
        destination = root / residual_settings.artifact_dir / panel
        destination.mkdir(parents=True, exist_ok=True)
        for split in config.request.submission_splits:
            frame_parts = []
            for path in trainer.masked_files(split):
                frame = pd.read_parquet(path, columns=list(MASKED_COLUMNS))
                frame_parts.append(study.predict_frame(full_model, full_corrector, frame))
            if not frame_parts:
                raise ProjectError(f"нет маскированных наблюдений {panel}/{split}")
            predictions = pd.concat(frame_parts, ignore_index=True)
            template = release_root / "task1" / panel / split / "sample_submission_state.csv"
            StateSubmissionWriter(config.layout.state_columns).write(template, predictions, destination / f"state_{split}.csv")
        return True


def _append_report(
    root: Path,
    settings: StateResidualSettings,
    panel: str,
    context_value: str,
    residual_value: str,
    choice: str,
) -> None:
    path = root / settings.artifact_dir / "holdout_state.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [{"panel": panel, "context": context_value, "residual": residual_value, "kind": choice}]
    )
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _queue_file(root: Path, settings: StateResidualSettings) -> Path:
    for relative in settings.queue_candidates:
        path = root / relative
        if path.is_file():
            return path
    raise ProjectError("нет файла очереди ни в одном из queue_candidates")


def _assemble(
    root: Path,
    config: ProjectConfig,
    context_settings: ContextSettings,
    residual_settings: StateResidualSettings,
    release_root: Path,
    panels: list[str],
    queue_file: Path,
) -> tuple[Path, int]:
    from trafficflowbench.baseline.pipeline import _concat_csv
    from trafficflowbench.submission.assembler import SubmissionAssembler

    version_dir = root / residual_settings.artifact_dir
    context_dir = root / context_settings.artifact_dir
    state_parts: list[Path] = []
    for split in config.request.submission_splits:
        parts: list[Path] = []
        for panel in panels:
            own = version_dir / panel / f"state_{split}.csv"
            previous = context_dir / panel / f"state_{split}.csv"
            chosen = own if own.is_file() else previous
            if not chosen.is_file():
                raise ProjectError(f"нет State {panel}/{split}")
            if chosen != own:
                copied = version_dir / panel / f"state_{split}.csv"
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(chosen, copied)
                chosen = copied
            parts.append(chosen)
        split_file = version_dir / f"state_{split}.csv"
        _concat_csv(parts, split_file)
        state_parts.append(split_file)
    state_file = version_dir / "state_submission.csv"
    _concat_csv(state_parts, state_file)
    odme = context_dir / "odme_submission.csv"
    upload = version_dir / "submission.csv"
    written = SubmissionAssembler(config.layout.merge).assemble(
        {"state": state_file, "queue": queue_file, "odme": odme},
        release_root / config.layout.submission_key,
        upload,
    )
    return upload, written


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    try:
        return RunStateResidualCommand(argv).run()
    except ProjectError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
