import json
from pathlib import Path
import subprocess


def test_compose_gateway_and_forwarded_proxy_trust_are_identical() -> None:
    repository = Path(__file__).parents[3]
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "infra/docker/compose.yaml",
            "--env-file",
            "infra/docker/.env.production.example",
            "config",
            "--format",
            "json",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    resolved = json.loads(result.stdout)
    gateway = resolved["networks"]["default"]["ipam"]["config"][0]["gateway"]
    api_environment = resolved["services"]["api"]["environment"]

    assert gateway == "172.30.0.1"
    assert api_environment["FORWARDED_ALLOW_IPS"] == gateway
