from pathlib import Path
import socket
import ssl
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import pytest


@pytest.fixture
def provider_url() -> Iterator[str]:
    repository = Path(__file__).resolve().parents[3]
    provider = repository / "infra/e2e/openai-provider/server.py"
    certificate = repository / "infra/e2e/tls/server.crt"
    certificate_authority = repository / "infra/e2e/tls/ca.crt"
    private_key = repository / "infra/e2e/tls/server.key"

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    process = subprocess.Popen(
        [
            sys.executable,
            str(provider),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--certificate",
            str(certificate),
            "--private-key",
            str(private_key),
            "--api-key",
            "e2e-api-key",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    url = f"https://127.0.0.1:{port}"
    deadline = time.monotonic() + 5
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr is not None else ""
                pytest.fail(f"E2E provider exited before readiness: {stderr}")
            try:
                response = httpx.get(
                    f"{url}/healthz",
                    verify=ssl.create_default_context(cafile=certificate_authority),
                    timeout=0.2,
                )
            except httpx.HTTPError:
                time.sleep(0.05)
                continue
            if response.status_code == 200:
                break
        else:
            pytest.fail("E2E provider did not become ready within 5 seconds")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def test_provider_accepts_multimodal_chat_completion(provider_url: str) -> None:
    response = httpx.post(
        f"{provider_url}/v1/chat/completions",
        headers={"Authorization": "Bearer e2e-api-key"},
        json={
            "model": "vision-e2e",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the test image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBORw0KGgo=",
                            },
                        },
                    ],
                }
            ],
        },
        verify=ssl.create_default_context(
            cafile=Path(__file__).resolve().parents[3] / "infra/e2e/tls/ca.crt"
        ),
        timeout=2,
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": "chatcmpl-grafy-e2e",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "vision-e2e",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The request contained text and one image.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 8,
            "total_tokens": 19,
        },
    }


def test_provider_rejects_completion_without_text(provider_url: str) -> None:
    response = httpx.post(
        f"{provider_url}/v1/chat/completions",
        headers={"Authorization": "Bearer e2e-api-key"},
        json={
            "model": "vision-e2e",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBORw0KGgo=",
                            },
                        }
                    ],
                }
            ],
        },
        verify=ssl.create_default_context(
            cafile=Path(__file__).resolve().parents[3] / "infra/e2e/tls/ca.crt"
        ),
        timeout=2,
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "E2E provider requires non-empty text content.",
            "type": "invalid_request_error",
            "param": "messages",
            "code": "missing_text",
        }
    }


def test_provider_rejects_completion_without_image(provider_url: str) -> None:
    response = httpx.post(
        f"{provider_url}/v1/chat/completions",
        headers={"Authorization": "Bearer e2e-api-key"},
        json={
            "model": "vision-e2e",
            "messages": [
                {
                    "role": "user",
                    "content": "Describe the test image.",
                }
            ],
        },
        verify=ssl.create_default_context(
            cafile=Path(__file__).resolve().parents[3] / "infra/e2e/tls/ca.crt"
        ),
        timeout=2,
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "E2E provider requires one inline image.",
            "type": "invalid_request_error",
            "param": "messages",
            "code": "missing_image",
        }
    }
