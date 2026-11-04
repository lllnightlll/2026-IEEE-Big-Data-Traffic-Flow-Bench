"""Исключения каркаса. Команда чтения ловит ProjectError и печатает причину без трассы."""


class ProjectError(Exception):
    """Ожидаемая ошибка постановки: путь, каталог коридоров или шаблон."""


class ConfigError(ProjectError):
    """Файл конфигурации отсутствует или не сходится с контрактом."""


class ReleaseNotFound(ProjectError):
    """Корень релиза не задан или не похож на распакованный kaggle_public."""


class CorridorNotFound(ProjectError):
    """Идентификатор панели отсутствует в corridors.json этого релиза."""


class TemplateNotFound(ProjectError):
    """Шаблон State, без которого команда чтения не может назвать число строк."""


class TemplateSchemaError(ProjectError):
    """Заголовок шаблона не содержит колонок, заданных в config/layout.yaml."""
