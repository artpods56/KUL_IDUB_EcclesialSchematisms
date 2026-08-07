from pathlib import Path
import re


def test_compose_gateway_and_forwarded_proxy_trust_are_identical() -> None:
    repository = Path(__file__).parents[3]
    compose = (repository / "infra/docker/compose.yaml").read_text()
    env_example = (repository / "infra/docker/.env.production.example").read_text()

    forwarded_allow_ips = re.search(
        r"^\s*FORWARDED_ALLOW_IPS:\s*(\$\{[^}]+\})\s*$",
        compose,
        re.MULTILINE,
    )
    gateway = re.search(
        r"^\s*gateway:\s*(\$\{[^}]+\})\s*$",
        compose,
        re.MULTILINE,
    )
    subnet = re.search(
        r"^\s*-\s+subnet:\s*(\$\{[^}]+\})\s*$",
        compose,
        re.MULTILINE,
    )
    assert forwarded_allow_ips is not None
    assert gateway is not None
    assert subnet is not None
    assert forwarded_allow_ips.group(1) == gateway.group(1)
    assert gateway.group(1) == "${NOTARIUS_DOCKER_GATEWAY:-172.30.0.1}"
    assert subnet.group(1) == "${NOTARIUS_DOCKER_SUBNET:-172.30.0.0/24}"

    env_values = dict(
        line.split("=", maxsplit=1)
        for line in env_example.splitlines()
        if line and not line.startswith("#") and "=" in line
    )
    assert env_values["NOTARIUS_DOCKER_GATEWAY"] == "172.30.0.1"
    assert env_values["NOTARIUS_DOCKER_SUBNET"] == "172.30.0.0/24"


def test_nginx_gateway_routes_api_and_mcp_to_same_upstream() -> None:
    repository = Path(__file__).parents[3]
    nginx = (repository / "infra/docker/gateway/nginx.conf").read_text()

    assert "upstream notarius_api" in nginx
    assert "server api:8000;" in nginx
    assert "upstream notarius_web" in nginx
    assert "server web:3000;" in nginx

    api_location = re.search(
        r"location\s+/api/\s*\{(.*?)\n\s*\}",
        nginx,
        re.DOTALL,
    )
    mcp_location = re.search(
        r"location\s+/mcp\s*\{(.*?)\n\s*\}",
        nginx,
        re.DOTALL,
    )
    assert api_location is not None, "gateway must expose /api/ to FastAPI"
    assert mcp_location is not None, "gateway must expose /mcp to mounted MCP"
    assert "proxy_pass http://notarius_api/;" in api_location.group(1)
    assert "proxy_pass http://notarius_api;" in mcp_location.group(1)
    assert "proxy_buffering off;" in api_location.group(1)
    assert "proxy_buffering off;" in mcp_location.group(1)
