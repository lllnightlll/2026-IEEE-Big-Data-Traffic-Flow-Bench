"""Команда обучения очереди. Правило v3 заменяется только после выигрыша на отложенных днях."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from trafficflowbench.cli.read_panel import _configure_stdio, repo_root_from
from trafficflowbench.config_loader import ProjectConfig
from trafficflowbench.data.reader import ReleaseRoot
from trafficflowbench.errors import ProjectError, ReleaseNotFound
from trafficflowbench.queue.model_settings import QueueModelSettings
from trafficflowbench.queue.settings import QueueSettings
from trafficflowbench.queue.study import QueueFit, QueuePanelStudy
from trafficflowbench.state.settings import ContextSettings


class RunQueueModelCommand:
    """Обучить коридоры Task 2 и, если отложенные дни это подтверждают, собрать новую сдачу."""

    def __init__(self, argv: list[str] | None = None) -> None:
        self.argv = argv if argv is not None else sys.argv[1:]

    def run(self) -> int:
        parser = argparse.ArgumentParser(description="Обучить очередь и собрать сдачу, если она лучше правила")
        parser.add_argument("--release-root", type=Path, default=None)
        parser.add_argument("--panel", default=None)
        parser.add_argument("--evaluate-only", action="store_true")
        parser.add_argument("--settings", type=Path, default=None, help="YAML обучения очереди, по умолчанию config/queue_model.yaml")
        args = parser.parse_args(self.argv)
        root = repo_root_from(Path(__file__))
        config = ProjectConfig.load(root, release_root_override=args.release_root)
        if config.request.release_root is None:
            raise ReleaseNotFound("путь к релизу не задан")
        release = ReleaseRoot(config.request.release_root)
        settings_path = args.settings if args.settings is not None else root / "config" / "queue_model.yaml"
        model_settings = QueueModelSettings.load(settings_path)
        rule = QueueSettings.load(root / "config" / "queue.yaml")
        panels = [args.panel] if args.panel else _task2_panels(release.path, config.request.submission_splits)
        version_dir = root / model_settings.artifact_dir
        version_dir.mkdir(parents=True, exist_ok=True)
        previous = _previous_scores(root, model_settings)
        fits: list[tuple[QueuePanelStudy, QueueFit]] = []
        for panel in panels:
            print(f"[queue-model] {panel}", flush=True)
            study = QueuePanelStudy(panel, release.path, config.scoring, model_settings, rule)
            fit = study.fit(previous_score=previous.get(panel))
            fits.append((study, fit))
            _print_summary(fit)
        report = pd.DataFrame([fit.summary for _, fit in fits])
        report_path = version_dir / "holdout_iou.csv"
        report.to_csv(report_path, index=False)
        print(f"holdout: {report_path}", flush=True)
        if args.evaluate_only or all(fit.kind in {"rule", "keep"} for _, fit in fits):
            print("submission: прежняя очередь остаётся, новая сдача не собрана", flush=True)
            return 0
        queue_file = self._queue_file(root, rule, model_settings, fits, config.request.submission_splits)
        context = ContextSettings.load(root / "config" / "context.yaml")
        upload, written = _assemble(config, context, model_settings, release.path, queue_file)
        print(f"submission: {upload}", flush=True)
        print(f"submission_rows: {written}", flush=True)
        return 0

    @staticmethod
    def _queue_file(
        root: Path,
        rule: QueueSettings,
        model_settings: QueueModelSettings,
        fits: list[tuple[QueuePanelStudy, QueueFit]],
        splits: tuple[str, ...],
    ) -> Path:
        base_dir = model_settings.base_artifact or rule.artifact_dir
        base = pd.read_csv(root / base_dir / "queue_submission.csv", dtype=str)
        base["window_id"] = base["window_id"].astype(str)
        replaced: list[pd.DataFrame] = []
        drop_ids: set[str] = set()
        for study, fit in fits:
            if fit.kind in {"rule", "keep"}:
                continue
            for split in splits:
                part = study.predict_split(fit, split)
                if part.empty:
                    continue
                replaced.append(part)
                drop_ids.update(part["window_id"].astype(str))
        kept = base[~base["window_id"].isin(drop_ids)].copy()
        kept["queue_pred"] = pd.to_numeric(kept["queue_pred"], errors="raise").astype(int)
        frames = [kept, *replaced] if replaced else [kept]
        combined = pd.concat(frames, ignore_index=True)
        if len(combined) != len(base):
            raise ProjectError(f"после замены модели строк {len(combined)}, в правиле было {len(base)}")
        if int(combined.duplicated(["window_id", "timestamp", "link_id"]).sum()):
            raise ProjectError("после замены модели появились дубли ключа очереди")
        output = root / model_settings.artifact_dir / "queue_submission.csv"
        combined.to_csv(output, index=False)
        print(f"queue_rows: {len(combined)}", flush=True)
        print(f"queue_ones: {int(combined['queue_pred'].sum())}", flush=True)
        return output


def _print_summary(fit: QueueFit) -> None:
    summary = fit.summary
    print(
        "  "
        f"{fit.kind}, порог {summary.get('threshold')}, "
        f"onset {summary.get('report_onset')}, ongoing {summary.get('report_ongoing')}",
        flush=True,
    )
    print(
        "  "
        f"persistence {summary.get('report_persistence')} "
        f"rule {summary.get('report_rule')} "
        f"model {summary.get('report_model')} "
        f"blend {summary.get('report_blend')} "
        f"split {summary.get('report_split', '')}",
        flush=True,
    )


def _previous_scores(root: Path, settings: QueueModelSettings) -> dict[str, float]:
    """Балл уже принятого выбора по коридору. Новый слой должен обойти его, а не только правило."""
    if not settings.previous_report:
        return {}
    path = root / settings.previous_report
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    if "panel" not in frame.columns or "kind" not in frame.columns:
        return {}
    column_of = {"model": "report_model", "blend": "report_blend", "rule": "report_rule", "split": "report_split"}
    scores: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        column = column_of.get(str(row.kind))
        if column is None or column not in frame.columns:
            continue
        value = getattr(row, column)
        if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
            continue
        scores[str(row.panel)] = float(value)
    return scores


def _task2_panels(release_root: Path, splits: tuple[str, ...]) -> list[str]:
    task_root = release_root / "task2"
    if not task_root.is_dir():
        raise ProjectError(f"нет каталога Task 2: {task_root}")
    return [
        panel.name
        for panel in sorted(path for path in task_root.iterdir() if path.is_dir())
        if any((panel / split / "sample_submission_queue.csv").is_file() for split in splits)
    ]


def _assemble(config: ProjectConfig, context: ContextSettings, settings: QueueModelSettings, release_root: Path, queue_file: Path) -> tuple[Path, int]:
    from trafficflowbench.submission.assembler import SubmissionAssembler

    previous = config.repo_root / context.artifact_dir
    upload = config.repo_root / settings.artifact_dir / "submission.csv"
    written = SubmissionAssembler(config.layout.merge).assemble(
        {
            "state": previous / "state_submission.csv",
            "queue": queue_file,
            "odme": previous / "odme_submission.csv",
        },
        release_root / config.layout.submission_key,
        upload,
    )
    return upload, written


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    try:
        return RunQueueModelCommand(argv).run()
    except ProjectError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
