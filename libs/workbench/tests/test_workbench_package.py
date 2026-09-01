import tomllib
from pathlib import Path


def test_workbench_package_is_app_owned() -> None:
    project = Path(__file__).resolve().parents[1]
    document = tomllib.loads((project / "pyproject.toml").read_text())

    assert document["project"]["name"] == "grafy-workbench"
    assert "grafy-core" in document["project"]["dependencies"]
    assert document["tool"]["uv"]["sources"]["grafy-core"] == {"workspace": True}
    assert not (project / "wheels").exists()
