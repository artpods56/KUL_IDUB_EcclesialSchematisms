from types import ModuleType
from unittest.mock import Mock

import pytest

from grafy_core.plugin_inspector import inspect_plugin
from grafy_plugin_text import TEXT

import grafy_core.plugin_inspector as plugin_inspector


def test_inspector_loads_a_real_converged_system_family_package() -> None:
    inspected = inspect_plugin("grafy_plugin_text.plugin:TEXT")

    assert inspected.catalog.slug == "builtin.text"
    assert inspected.catalog.title == TEXT.title


def test_inspector_defaults_to_the_fixed_workspace_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("grafy_plugin")
    module.PLUGIN = TEXT  # type: ignore[attr-defined]
    importer = Mock(return_value=module)
    monkeypatch.setattr(plugin_inspector, "import_module", importer)

    inspected = inspect_plugin()

    assert inspected.catalog.slug == "builtin.text"
    importer.assert_called_once_with("grafy_plugin")
