"""Probe prompt-sensitive Gemma structured-output failures."""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeCase:
    name: str
    prompt: str
    schema: dict[str, object]


@dataclass(frozen=True)
class ProbeResult:
    case: str
    attempt: int
    status: int
    seconds: float
    content: str | None
    error: str | None


def send_probe(
    url: str,
    model: str,
    case: ProbeCase,
    attempt: int,
    timeout_seconds: int,
) -> ProbeResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": case.prompt}],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": case.name,
                "strict": True,
                "schema": case.schema,
            },
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = str(body["choices"][0]["message"]["content"])
        json.loads(content)
        return ProbeResult(
            case=case.name,
            attempt=attempt,
            status=response.status,
            seconds=round(time.monotonic() - started, 3),
            content=content,
            error=None,
        )
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        return ProbeResult(
            case=case.name,
            attempt=attempt,
            status=error.code,
            seconds=round(time.monotonic() - started, 3),
            content=None,
            error=body,
        )
    except (KeyError, OSError, ValueError) as error:
        return ProbeResult(
            case=case.name,
            attempt=attempt,
            status=0,
            seconds=round(time.monotonic() - started, 3),
            content=None,
            error=str(error),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()

    if args.attempts < 1:
        parser.error("attempts must be positive")

    integer_schema: dict[str, object] = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    classification_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["label", "confidence"],
        "additionalProperties": False,
    }
    cases = [
        ProbeCase(
            name="integer_ambiguous",
            prompt="Oblicz 17 razy 3 i zwróć wynik jako JSON z polem value.",
            schema=integer_schema,
        ),
        ProbeCase(
            name="integer_no_markdown",
            prompt=(
                "Oblicz 17 razy 3. Zwróć wyłącznie JSON z polem value. "
                "Nie używaj Markdown ani bloków kodu. Zacznij odpowiedź znakiem {."
            ),
            schema=integer_schema,
        ),
        ProbeCase(
            name="classification_ambiguous",
            prompt="Sklasyfikuj tekst testowy i zwróć wynik jako JSON.",
            schema=classification_schema,
        ),
        ProbeCase(
            name="classification_no_markdown",
            prompt=(
                "Sklasyfikuj tekst testowy. Zwróć wyłącznie JSON z polami label "
                "i confidence. Nie używaj Markdown ani bloków kodu."
            ),
            schema=classification_schema,
        ),
    ]

    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    failed = False
    for case in cases:
        for attempt in range(1, args.attempts + 1):
            result = send_probe(
                url,
                args.model,
                case,
                attempt,
                args.timeout_seconds,
            )
            print(json.dumps(result.__dict__, ensure_ascii=False, sort_keys=True))
            failed = failed or result.status != 200
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
