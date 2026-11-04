"""Склейка трёх файлов сдачи на ключ организатора.

Официальный merge_submissions.py дыры заполняет нулём и только предупреждает.
Здесь дыра отвергает файл: неполный блок задачи на Kaggle обнуляет весь блок.
Неиспользуемые колонки строки заполняются нулём, потому что платформа не принимает пустую ячейку.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trafficflowbench.config_loader import MergeContract, MergeTask
from trafficflowbench.errors import SubmissionRejected


class TaskTable:
    """Файл одной задачи, уже проверенный и проиндексированный своим ключом."""

    def __init__(self, spec: MergeTask, frame: pd.DataFrame) -> None:
        self.spec = spec
        self.frame = frame

    @classmethod
    def load(cls, path: Path, spec: MergeTask) -> TaskTable:
        if not path.is_file():
            raise SubmissionRejected(f"нет файла задачи {spec.name}: {path}")
        frame = pd.read_csv(path)
        required = list(spec.keys) + list(spec.values)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise SubmissionRejected(f"{path} не содержит колонки {missing}")
        frame = frame[required].copy()
        for column in spec.keys:
            if column == "timestamp":
                frame[column] = pd.to_datetime(frame[column], utc=True)
            else:
                frame[column] = frame[column].astype(str)
        for column in spec.values:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        cls._reject_bad_values(path, spec, frame)
        duplicated = int(frame.duplicated(list(spec.keys)).sum())
        if duplicated:
            raise SubmissionRejected(f"{path} содержит {duplicated} дублей ключа {list(spec.keys)}")
        return cls(spec, frame.set_index(list(spec.keys)))

    @staticmethod
    def _reject_bad_values(path: Path, spec: MergeTask, frame: pd.DataFrame) -> None:
        """Отсечь значения, которые платформа или оценщик всё равно забракуют."""
        for column in spec.values:
            values = frame[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise SubmissionRejected(f"{path}: в {column} есть пустые или неконечные значения")
            if spec.name == "queue" and not np.isin(values, (0.0, 1.0)).all():
                raise SubmissionRejected(f"{path}: queue_pred может быть только 0 или 1")
            if spec.name == "odme" and (values < 0).any():
                raise SubmissionRejected(f"{path}: path_flow должен быть неотрицательным")


class SubmissionAssembler:
    """Потоковая посадка трёх таблиц на submission_key.csv. Успех — только полное покрытие ключа."""

    def __init__(self, contract: MergeContract) -> None:
        self.contract = contract

    def assemble(self, tables: dict[str, Path], key: Path, output: Path) -> int:
        loaded = {
            task.name: TaskTable.load(tables[task.name], task)
            for task in self.contract.tasks
        }
        missing_names = [task.name for task in self.contract.tasks if task.name not in tables]
        if missing_names:
            raise SubmissionRejected(f"для склейки не переданы задачи: {missing_names}")
        if not key.is_file():
            raise SubmissionRejected(f"нет submission_key.csv: {key}")

        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(output.suffix + ".partial")
        gaps = {task.name: 0 for task in self.contract.tasks}
        written = 0
        first = True
        accepted = False
        try:
            reader = pd.read_csv(key, chunksize=self.contract.chunk_rows, dtype=str)
            for chunk in reader:
                self._require_key_columns(chunk, key)
                if "timestamp" in chunk.columns:
                    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce")
                out = pd.DataFrame(
                    {
                        "submission_id": pd.to_numeric(chunk["submission_id"], errors="raise").astype("int64"),
                        "task": chunk["task"].astype(str),
                    }
                )
                for column in self.contract.output_columns:
                    if column not in ("submission_id", "task"):
                        out[column] = 0.0
                for task in self.contract.tasks:
                    rows = out["task"].eq(task.name)
                    if not bool(rows.any()):
                        continue
                    wanted = pd.MultiIndex.from_frame(chunk.loc[rows, list(task.keys)])
                    found = loaded[task.name].frame.reindex(wanted)
                    for column in task.values:
                        values = found[column].to_numpy()
                        gaps[task.name] += int(pd.isna(values).sum())
                        # Дыры остаются нулями только внутри черновика. Если они есть, черновик удаляется.
                        filled = pd.Series(values, index=out.index[rows]).fillna(0.0)
                        out.loc[rows, column] = filled.to_numpy()
                out.loc[:, list(self.contract.output_columns)].to_csv(
                    partial,
                    mode="w" if first else "a",
                    header=first,
                    index=False,
                )
                first = False
                written += len(out)
            open_gaps = {name: count for name, count in gaps.items() if count}
            if open_gaps:
                detail = ", ".join(f"{name}: {count}" for name, count in open_gaps.items())
                raise SubmissionRejected(
                    f"склейка неполная, файл загрузки не сохранён. Строки без значения: {detail}. "
                    "Неполный блок задачи обнуляет её целиком."
                )
            if written == 0:
                raise SubmissionRejected(f"ключ сдачи пуст: {key}")
            partial.replace(output)
            accepted = True
            return written
        finally:
            if not accepted and partial.exists():
                partial.unlink()

    def _require_key_columns(self, chunk: pd.DataFrame, key: Path) -> None:
        needed = {"submission_id", "task"}
        for task in self.contract.tasks:
            needed.update(task.keys)
        missing = sorted(needed - set(chunk.columns))
        if missing:
            raise SubmissionRejected(f"{key} не содержит колонки {missing}")
