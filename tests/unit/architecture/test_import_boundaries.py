from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

FORBIDDEN_CORE_IMPORTS = (
    "fastapi",
    "faststream",
    "nats",
    "sqlalchemy",
    "notarius_api",
    "notarius_dagster",
    "notarius_llm",
    "notarius_messaging",
    "notarius_persistence",
    "notarius_schematisms",
    "notarius_storage",
    "notarius_worker",
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
