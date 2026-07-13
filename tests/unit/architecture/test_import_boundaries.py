import tomllib
from pathlib import Path
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[3]

FORBIDDEN_CORE_IMPORTS = (
    "fastapi",
    "mistralai",
    "notarius_api",
    "notarius_plugin_ocr",
    "notarius_storage",
)
FORBIDDEN_OCR_PLUGIN_IMPORTS = (
    "notarius_api",
    "notarius_storage",
)
FORBIDDEN_API_PLUGIN_IMPORTS = (
    "mistralai",
    "notarius_plugin_ocr",
)
LEGACY_NAMESPACE = "proto" + "type"


def test_ocr_sdk_dependency_is_owned_by_the_optional_plugin() -> None:
    root_document = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    api_document = tomllib.loads(
        (REPO_ROOT / "apps/api/pyproject.toml").read_text()
    )
    core_document = tomllib.loads(
        (REPO_ROOT / "libs/core/pyproject.toml").read_text()
    )
    plugin_document = tomllib.loads(
        (REPO_ROOT / "plugins/ocr/pyproject.toml").read_text()
    )

    root_project = cast(dict[str, object], root_document["project"])
    api_project = cast(dict[str, object], api_document["project"])
    core_project = cast(dict[str, object], core_document["project"])
    plugin_project = cast(dict[str, object], plugin_document["project"])

    root_dependencies = cast(list[str], root_project["dependencies"])
    root_extras = cast(
        dict[str, list[str]], root_project["optional-dependencies"]
    )
    api_dependencies = cast(list[str], api_project["dependencies"])
    core_dependencies = cast(list[str], core_project["dependencies"])
    plugin_dependencies = cast(list[str], plugin_project["dependencies"])

    assert not any(
        requirement.startswith("notarius-plugin-ocr")
        for requirement in root_dependencies
    )
    assert not any(
        requirement.startswith("mistralai")
        for requirement in root_dependencies
    )
    assert root_extras["ocr"] == ["notarius-plugin-ocr"]

    for dependencies in (api_dependencies, core_dependencies):
        assert not any(
            requirement.startswith("notarius-plugin-ocr")
            for requirement in dependencies
        )
        assert not any(
            requirement.startswith("mistralai")
            for requirement in dependencies
        )

    assert any(
        requirement.startswith("mistralai")
        for requirement in plugin_dependencies
    )


def test_core_does_not_import_outer_layers_or_domain_adapters() -> None:
    core_root = REPO_ROOT / "libs/core/src/notarius_core"
    offenders: list[str] = []

    for path in core_root.rglob("*.py"):
        text = path.read_text()
        for forbidden in FORBIDDEN_CORE_IMPORTS:
            if f"import {forbidden}" in text or f"from {forbidden}" in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {forbidden}")

    assert offenders == []


def test_api_host_does_not_import_optional_plugin_implementations() -> None:
    api_root = REPO_ROOT / "apps/api/src/notarius_api"
    offenders: list[str] = []

    for path in api_root.rglob("*.py"):
        text = path.read_text()
        for forbidden in FORBIDDEN_API_PLUGIN_IMPORTS:
            if f"import {forbidden}" in text or f"from {forbidden}" in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {forbidden}")

    assert offenders == []


def test_ocr_plugin_depends_on_core_ports_not_outer_layers() -> None:
    plugin_root = REPO_ROOT / "plugins/ocr/src/notarius_plugin_ocr"
    offenders: list[str] = []

    for path in plugin_root.rglob("*.py"):
        text = path.read_text()
        for forbidden in FORBIDDEN_OCR_PLUGIN_IMPORTS:
            if f"import {forbidden}" in text or f"from {forbidden}" in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {forbidden}")

    assert offenders == []


def test_retained_python_sources_do_not_use_legacy_namespace() -> None:
    source_roots = (
        REPO_ROOT / "libs/core/src/notarius_core",
        REPO_ROOT / "plugins/ocr/src/notarius_plugin_ocr",
        REPO_ROOT / "apps/api/src/notarius_api",
    )
    offenders: list[str] = []

    for source_root in source_roots:
        for path in source_root.rglob("*.py"):
            relative_path = path.relative_to(REPO_ROOT)
            if LEGACY_NAMESPACE in relative_path.as_posix().lower():
                offenders.append(str(relative_path))
                continue
            if LEGACY_NAMESPACE in path.read_text().lower():
                offenders.append(str(relative_path))

    assert offenders == []
