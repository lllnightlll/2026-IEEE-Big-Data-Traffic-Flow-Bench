"""Каталог направленных коридоров.

Список панелей не задаётся в коде. Его источник — config/corridors.json
распакованного релиза, тот же файл, которым пользуются скрипты организаторов.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from trafficflowbench.errors import ConfigError, CorridorNotFound


@dataclass(frozen=True)
class Corridor:
    """Одна направленная панель. Имена полей совпадают с ключами corridors.json."""

    corridor_id: str
    family_id: str
    district: int
    freeway_number: int
    direction: str


@dataclass(frozen=True)
class CorridorCatalog:
    """Все панели одного релиза в порядке файла."""

    source: Path
    corridors: tuple[Corridor, ...]

    @classmethod
    def load(cls, path: Path) -> CorridorCatalog:
        if not path.is_file():
            raise ConfigError(f"в релизе нет каталога коридоров: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        panels = document.get("panels") if isinstance(document, dict) else None
        if not isinstance(panels, list) or not panels:
            raise ConfigError(f"{path} не содержит список panels")
        corridors: list[Corridor] = []
        for index, raw in enumerate(panels):
            if not isinstance(raw, dict):
                raise ConfigError(f"{path}: panels[{index}] должен быть объектом")
            try:
                corridors.append(
                    Corridor(
                        corridor_id=str(raw["corridor_id"]),
                        family_id=str(raw["family_id"]),
                        district=int(raw["district"]),
                        freeway_number=int(raw["freeway_number"]),
                        direction=str(raw["direction"]),
                    )
                )
            except KeyError as exc:
                raise ConfigError(f"{path}: у панели {index} нет поля {exc.args[0]}") from exc
        return cls(source=path, corridors=tuple(corridors))

    def ids(self) -> tuple[str, ...]:
        return tuple(corridor.corridor_id for corridor in self.corridors)

    def require(self, corridor_id: str) -> Corridor:
        """Вернуть панель или отказаться. Чтение каталога по чужому имени не начинается."""
        for corridor in self.corridors:
            if corridor.corridor_id == corridor_id:
                return corridor
        known = ", ".join(self.ids()) or "(пусто)"
        raise CorridorNotFound(
            f"панель {corridor_id} отсутствует в {self.source}. В этом релизе есть: {known}"
        )
