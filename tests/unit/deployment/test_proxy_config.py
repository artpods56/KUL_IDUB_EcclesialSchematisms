from pathlib import Path
import re


def test_compose_trusts_configured_docker_subnet_for_forwarded_headers() -> None:
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
    assert forwarded_allow_ips.group(1) == subnet.group(1)
    assert forwarded_allow_ips.group(1) != gateway.group(1)
    assert gateway.group(1) == "${GRAFY_DOCKER_GATEWAY:-172.30.0.1}"
    assert subnet.group(1) == "${GRAFY_DOCKER_SUBNET:-172.30.0.0/24}"

    env_values = dict(
        line.split("=", maxsplit=1)
        for line in env_example.splitlines()
        if line and not line.startswith("#") and "=" in line
    )
    assert env_values["GRAFY_DOCKER_GATEWAY"] == "172.30.0.1"
    assert env_values["GRAFY_DOCKER_SUBNET"] == "172.30.0.0/24"


def test_compose_injects_complete_oidc_configuration_and_requires_keys() -> None:
    repository = Path(__file__).parents[3]
    compose = (repository / "infra/docker/compose.yaml").read_text()
    env_example = (repository / "infra/docker/.env.production.example").read_text()

    required_nonempty = (
        "GRAFY_OIDC_ISSUER",
        "GRAFY_OIDC_CLIENT_ID",
        "GRAFY_OIDC_AUTH_WRAPPING_KEY",
        "GRAFY_CREDENTIAL_ENCRYPTION_KEY",
        "GRAFY_COMMAND_HMAC_KEY",
    )
    for variable in required_nonempty:
        assert re.search(
            rf"^\s*{variable}:\s*\$\{{{variable}:\?[^}}]+\}}\s*$",
            compose,
            re.MULTILINE,
        ), f"{variable} must fail Compose interpolation when unset or empty"

    assert re.search(
        r"^\s*GRAFY_OIDC_CLIENT_SECRET:\s*$",
        compose,
        re.MULTILINE,
    )
    assert (
        "GRAFY_OIDC_ALLOWED_SIGNING_ALGORITHMS: "
        '${GRAFY_OIDC_ALLOWED_SIGNING_ALGORITHMS:-["RS256"]}' in compose
    )
    assert (
        "GRAFY_OIDC_AUTH_WRAPPING_KEY_VERSION: "
        "${GRAFY_OIDC_AUTH_WRAPPING_KEY_VERSION:-1}" in compose
    )

    env_values = dict(
        line.split("=", maxsplit=1)
        for line in env_example.splitlines()
        if line and not line.startswith("#") and "=" in line
    )
    for variable in required_nonempty:
        assert variable in env_values
    assert env_values["GRAFY_OIDC_ALLOWED_SIGNING_ALGORITHMS"] == '["RS256"]'
    assert env_values["GRAFY_OIDC_AUTH_WRAPPING_KEY_VERSION"] == "1"


def test_compose_injects_staged_upload_hard_limit() -> None:
    repository = Path(__file__).parents[3]
    compose = (repository / "infra/docker/compose.yaml").read_text()
    env_example = (repository / "infra/docker/.env.production.example").read_text()

    assert (
        "GRAFY_STAGED_UPLOAD_MAX_BYTES: "
        "${GRAFY_STAGED_UPLOAD_MAX_BYTES:-67108864}" in compose
    )
    assert "GRAFY_STAGED_UPLOAD_MAX_BYTES=67108864" in env_example


def test_compose_data_volume_has_an_explicit_migration_override() -> None:
    repository = Path(__file__).parents[3]
    compose = (repository / "infra/docker/compose.yaml").read_text()
    env_example = (repository / "infra/docker/.env.production.example").read_text()

    assert "name: ${GRAFY_DATA_VOLUME:-grafy-data}" in compose
    assert "GRAFY_DATA_VOLUME=grafy-data" in env_example


def test_nginx_gateway_routes_web_api_and_mcp_to_same_origin_upstreams() -> None:
    repository = Path(__file__).parents[3]
    nginx = (repository / "infra/docker/gateway/nginx.conf").read_text()

    assert "upstream grafy_api" in nginx
    assert "server api:8000;" in nginx
    assert "upstream grafy_web" in nginx
    assert "server web:3000;" in nginx
    assert "listen 8080;" in nginx
    assert "client_max_body_size 65m;" in nginx

    root_location = re.search(
        r"location\s+/\s*\{(.*?)\n\s*\}",
        nginx,
        re.DOTALL,
    )
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
    assert root_location is not None, "gateway must expose / to Next.js"
    assert api_location is not None, "gateway must expose /api/ to FastAPI"
    assert mcp_location is not None, "gateway must expose /mcp to mounted MCP"
    assert "proxy_pass http://grafy_web;" in root_location.group(1)
    assert "proxy_pass http://grafy_api/;" in api_location.group(1)
    assert "proxy_pass http://grafy_api;" in mcp_location.group(1)
    assert "proxy_buffering off;" in api_location.group(1)
    assert "proxy_buffering off;" in mcp_location.group(1)


def test_compose_publishes_loopback_gateway_and_keeps_api_web_internal() -> None:
    repository = Path(__file__).parents[3]
    compose = (repository / "infra/docker/compose.yaml").read_text()
    dockerfile = (repository / "infra/docker/api.Dockerfile").read_text()

    assert re.search(r"^  gateway:\s*$", compose, re.MULTILINE)
    assert "./gateway/nginx.conf:/etc/nginx/nginx.conf:ro" in compose
    assert (
        "${GRAFY_BIND_ADDRESS:-127.0.0.1}:${GRAFY_GATEWAY_PORT:-8080}:8080"
        in compose
    )
    assert 'GRAFY_REQUIRE_SINGLE_API_OWNER: "true"' in compose
    assert 'WEB_CONCURRENCY: "1"' in compose

    # API and web stay on the Compose network; only gateway (and Prefect SSH
    # dashboard) publish host ports.
    api_block = re.search(
        r"^  api:\n(.*?)(?=^  [a-z]|\Z)",
        compose,
        re.MULTILINE | re.DOTALL,
    )
    web_block = re.search(
        r"^  web:\n(.*?)(?=^  [a-z]|\Z)",
        compose,
        re.MULTILINE | re.DOTALL,
    )
    assert api_block is not None
    assert web_block is not None
    assert not re.search(r"^\s+ports:\s*$", api_block.group(1), re.MULTILINE)
    assert not re.search(r"^\s+ports:\s*$", web_block.group(1), re.MULTILINE)
    assert 'expose:\n      - "8000"' in api_block.group(1)
    assert 'expose:\n      - "3000"' in web_block.group(1)

    assert "--workers" not in dockerfile
    assert "grafy_api.main:app" in dockerfile


def test_production_api_does_not_install_untrusted_sql_in_process() -> None:
    repository = Path(__file__).parents[3]
    dockerfile = (repository / "infra/docker/api.Dockerfile").read_text()

    production_target = dockerfile.split("FROM source AS api-plugins", maxsplit=1)[1]
    production_target = production_target.split("FROM source AS api", maxsplit=1)[0]
    assert "--extra gis --extra llm --extra ocr" in production_target
    assert "--extra sql" not in production_target


def test_compose_api_healthcheck_uses_dependency_readiness() -> None:
    repository = Path(__file__).parents[3]
    compose = (repository / "infra/docker/compose.yaml").read_text()
    api_block = re.search(
        r"^  api:\n(.*?)(?=^  [a-z]|\Z)",
        compose,
        re.MULTILINE | re.DOTALL,
    )

    assert api_block is not None
    assert "http://127.0.0.1:8000/ready" in api_block.group(1)
    assert "http://127.0.0.1:8000/health" not in api_block.group(1)
