"""База v1: профили и решатели организаторов, затем локальные S_state и S_link.

Склейка в файл загрузки происходит только когда в прогоне все панели каталога.
Один коридор годится для сверки с опубликованным 0.6252, но не для отправки:
недостающие панели обнулили бы задачу целиком.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trafficflowbench.config_loader import ProjectConfig
from trafficflowbench.data.catalog import CorridorCatalog
from trafficflowbench.data.reader import ReleaseRoot
from trafficflowbench.errors import OrganizerFailed, ReleaseNotFound
from trafficflowbench.external.organizer import OrganizerRunner, ScoreSheet
from trafficflowbench.submission.assembler import SubmissionAssembler


class BaselinePipeline:
    """Три строителя, два локальных балла и, при полном наборе панелей, один CSV."""

    def __init__(self, config: ProjectConfig, python: Path) -> None:
        self.config = config
        self.runner = OrganizerRunner(python, config.upstream)

    def run(self, panels: list[str]) -> str:
        if self.config.request.release_root is None:
            raise ReleaseNotFound(
                "путь к релизу не задан. Запишите каталог kaggle_public в config/release.local.yaml, "
                "в TFB_RELEASE_ROOT или передайте --release-root."
            )
        root = ReleaseRoot(self.config.request.release_root)
        catalog = CorridorCatalog.load(root.path / self.config.layout.corridors_catalog)
        for panel in panels:
            catalog.require(panel)
        if not panels:
            raise ReleaseNotFound("список панелей пуст. Укажите --panel или --all-panels.")

        artifact_dir = self.config.repo_root / self.config.request.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        score_split = self.config.scoring.published_state_split
        panel_args = self.runner.panel_arguments(panels)

        state_train = artifact_dir / f"state_{score_split}.csv"
        self.runner.run(
            "state_build",
            ["--release-root", str(root.path), "--split", score_split, "--output", str(state_train), *panel_args],
        )
        state_score_path = artifact_dir / f"state_score_{score_split}.csv"
        self.runner.run(
            "state_score",
            [
                "--submission", str(state_train),
                "--release-root", str(root.path),
                "--split", score_split,
                "--output", str(state_score_path),
                *panel_args,
            ],
        )
        state_sheet = ScoreSheet(state_score_path)
        self._check_published_state(state_sheet, panels, score_split)
        state_overall = state_sheet.value("overall", "S_state")
        submission_splits = self.config.request.submission_splits
        state_parts: list[Path] = []
        queue_parts: list[Path] = []
        odme_parts: list[Path] = []
        for submit_split in submission_splits:
            state_part = artifact_dir / f"state_{submit_split}.csv"
            self.runner.run(
                "state_build",
                [
                    "--release-root", str(root.path),
                    "--split", submit_split,
                    "--output", str(state_part),
                    *panel_args,
                ],
            )
            queue_part = artifact_dir / f"queue_{submit_split}.csv"
            # У строителя очереди нет отбора панели: история окон читается целиком для сплита.
            self.runner.run(
                "queue_build",
                ["--release-root", str(root.path), "--split", submit_split, "--output", str(queue_part)],
            )
            odme_root = artifact_dir / f"odme_{submit_split}"
            self.runner.run(
                "odme_build",
                [
                    "--release-root", str(root.path),
                    "--split", submit_split,
                    "--output-root", str(odme_root),
                    *panel_args,
                ],
            )
            state_parts.append(state_part)
            queue_parts.append(queue_part)
            odme_parts.append(odme_root / "baseline_submission.csv")

        # Локальный S_link смотрим на validation: private без эталонных счётчиков в открытом пакете не показательнее.
        link_split = "validation" if "validation" in submission_splits else submission_splits[0]
        link_index = submission_splits.index(link_split)
        odme_score_path = artifact_dir / f"odme_score_{link_split}.csv"
        self.runner.run(
            "odme_score",
            [
                "--submission", str(odme_parts[link_index]),
                "--release-root", str(root.path),
                "--split", link_split,
                "--output", str(odme_score_path),
                *panel_args,
            ],
        )
        link_overall = ScoreSheet(odme_score_path).value("overall", "S_link")

        lines = [
            f"S_state({score_split}): {state_overall:.6f}",
            f"S_link({link_split}): {link_overall:.6f}",
        ]
        if set(panels) == set(catalog.ids()):
            state_file = artifact_dir / "state_submission.csv"
            queue_file = artifact_dir / "queue_submission.csv"
            odme_file = artifact_dir / "odme_submission.csv"
            _concat_csv(state_parts, state_file)
            _concat_csv(queue_parts, queue_file)
            _concat_csv(odme_parts, odme_file)
            upload = artifact_dir / "submission.csv"
            written = SubmissionAssembler(self.config.layout.merge).assemble(
                {"state": state_file, "queue": queue_file, "odme": odme_file},
                root.path / self.config.layout.submission_key,
                upload,
            )
            lines.append(f"submission: {upload}")
            lines.append(f"submission_rows: {written}")
        else:
            lines.append(
                "Файл загрузки не собран: в прогоне не все панели каталога. "
                "Неполный блок на Kaggle обнуляет задачу. Для полного CSV запустите с --all-panels."
            )
        return "\n".join(lines) + "\n"

    def _check_published_state(self, sheet: ScoreSheet, panels: list[str], split: str) -> None:
        """Сверить панель, для которой организатор опубликовал число. На остальных панелях ожидания нет."""
        scoring = self.config.scoring
        if split != scoring.published_state_split:
            return
        if scoring.published_state_panel not in panels:
            return
        actual = sheet.value("panel", "S_state", panel=scoring.published_state_panel)
        allowed = scoring.published_state_tolerance
        if abs(actual - scoring.published_state_score) > allowed:
            raise OrganizerFailed(
                f"S_state панели {scoring.published_state_panel} на {split} равен {actual:.6f}, "
                f"опубликованная база {scoring.published_state_score:.4f} с допуском {allowed}."
            )


def _concat_csv(parts: list[Path], output: Path) -> None:
    """Склеить сдачи нескольких сплитов, не держа оба месяца в памяти целиком."""
    first = True
    output.parent.mkdir(parents=True, exist_ok=True)
    for part in parts:
        if not part.is_file():
            raise OrganizerFailed(f"нет части сдачи: {part}")
        for chunk in pd.read_csv(part, chunksize=500_000):
            chunk.to_csv(output, mode="w" if first else "a", header=first, index=False)
            first = False
    if first:
        raise OrganizerFailed(f"нечего склеивать в {output}")
