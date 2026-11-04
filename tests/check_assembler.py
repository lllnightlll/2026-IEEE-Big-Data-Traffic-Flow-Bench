"""Склейка принимает полный ключ и отказывается, если хотя бы одна строка задачи не нашлась."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trafficflowbench.config_loader import ProjectConfig  # noqa: E402
from trafficflowbench.errors import SubmissionRejected  # noqa: E402
from trafficflowbench.external.organizer import OrganizerRunner  # noqa: E402
from trafficflowbench.submission.assembler import SubmissionAssembler  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "merge"


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    config = ProjectConfig.load(REPO)
    assembler = SubmissionAssembler(config.layout.merge)
    output = FIXTURE / "submission.csv"
    if output.exists():
        output.unlink()
    written = assembler.assemble(
        {
            "state": FIXTURE / "state.csv",
            "queue": FIXTURE / "queue.csv",
            "odme": FIXTURE / "odme.csv",
        },
        FIXTURE / "submission_key.csv",
        output,
    )
    expect(written == 3, f"ожидалось 3 строки склейки, получено {written}")
    text = output.read_text(encoding="utf-8")
    for line in text.splitlines()[1:]:
        fields = line.split(",")
        expect(len(fields) == 6, f"в строке сдачи не шесть колонок: {line}")
        expect(all(field != "" for field in fields), f"пустая ячейка в {line}")
    output.unlink()

    rejected = FIXTURE / "rejected.csv"
    try:
        assembler.assemble(
            {
                "state": FIXTURE / "state.csv",
                "queue": FIXTURE / "queue_gap.csv",
                "odme": FIXTURE / "odme.csv",
            },
            FIXTURE / "submission_key.csv",
            rejected,
        )
    except SubmissionRejected:
        pass
    else:
        raise SystemExit("дыра в очереди должна отвергать склейку")
    expect(not rejected.exists(), "отвергнутая сдача не должна остаться на диске")
    expect(not rejected.with_suffix(rejected.suffix + ".partial").exists(), "черновик отвергнутой сдачи должен быть удалён")

    arguments = OrganizerRunner.panel_arguments(["D12_I5_N"])
    expect(arguments == ["--panel", "D12_I5_N"], "флаг панели строится из переданного списка")
    print("assembler checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
