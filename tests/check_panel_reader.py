"""Проверка каркаса на маленьком релизе той же раскладки, что и kaggle_public.

Образец намеренно не содержит немаскированный слой у validation.
Читатель должен вернуть число строк State и не поднять исключение из-за этого отсутствия.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trafficflowbench.cli.read_panel import ReadPanelCommand  # noqa: E402
from trafficflowbench.config_loader import ProjectConfig  # noqa: E402
from trafficflowbench.data.reader import PanelReader, ReleaseRoot  # noqa: E402
from trafficflowbench.errors import CorridorNotFound, ProjectError  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "mini_release"


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    config = ProjectConfig.load(REPO, release_root_override=FIXTURE)
    reader = PanelReader(ReleaseRoot(FIXTURE), config.layout)
    expect(len(reader.catalog.ids()) == 10, "каталог должен прийти из corridors.json и содержать десять панелей")

    validation = reader.read("D12_I5_N", "validation")
    expect(validation.state_template.rows == 4, f"ожидалось 4 строки State, получено {validation.state_template.rows}")
    expect(validation.queue_template.rows == 2, "шаблон Queue образца должен содержать 2 строки")
    expect(validation.odme_template.rows == 2, "шаблон ODME образца должен содержать 2 строки")
    expect(validation.unmasked.present is False, "у validation не должно быть немаскированного слоя")
    expect(validation.unmasked.file_count == 0, "отсутствующий слой не содержит файлов")
    expect(validation.masked.present is True, "маскированный слой validation в образце есть")
    expect(validation.masked.file_count == 1, "в образце один маскированный parquet")

    train = reader.read("D12_I5_N", "train")
    expect(train.state_template.rows == 2, "у train в образце две строки State")
    expect(train.unmasked.present is True and train.unmasked.file_count == 1, "у train немаскированный слой есть")
    expect(train.queue_template.present is False, "у train в образце шаблон Queue не обязателен")

    try:
        reader.read("NO_SUCH_PANEL", "validation")
    except CorridorNotFound:
        pass
    else:
        raise SystemExit("чужая панель должна отвергаться каталогом")

    command = ReadPanelCommand(
        ["--release-root", str(FIXTURE), "--panel", "D12_I5_N", "--split", "validation"]
    )
    expect(command.run() == 0, "команда чтения validation должна завершаться нулём")

    missing = ReadPanelCommand([])
    try:
        code = missing.run()
    except ProjectError:
        code = 1
    expect(code == 1, "без пути к релизу команда должна отказаться, а не искать каталог в коде")
    print("fixture checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
