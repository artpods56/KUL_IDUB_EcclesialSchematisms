"""Run repeatable concurrent plain and structured-output requests against Gemma."""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_payload(
    mode: str,
    prompt_chars: int,
    request_label: str,
) -> dict[str, object]:
    source = "Archiwalny dokument zawiera dane do klasyfikacji. "
    repetitions = (prompt_chars // len(source)) + 1
    document = (source * repetitions)[:prompt_chars]
    payload: dict[str, object] = {
        "model": "google/gemma-4-31B-it",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Przeczytaj tekst i odpowiedz zgodnie z instrukcją.\n\n"
                    f"Identyfikator testu: {request_label}\n\n"
                    f"Tekst:\n{document}"
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": 96,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if mode == "plain":
        payload["messages"] = [
            {
                "role": "user",
                "content": (
                    "Odpowiedz dokładnie jednym słowem OK po przeczytaniu tekstu.\n\n"
                    f"Identyfikator testu: {request_label}\n\n"
                    f"Tekst:\n{document}"
                ),
            }
        ]
    else:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "document_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["label", "confidence"],
                    "additionalProperties": False,
                },
            },
        }
    return payload


def send_request(
    url: str,
    payload: dict[str, object],
    barrier: threading.Barrier,
    timeout_seconds: int,
) -> dict[str, object]:
    barrier.wait()
    started = time.monotonic()
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_response = response.read().decode("utf-8")
        result = json.loads(raw_response)
        content = result["choices"][0]["message"]["content"]
        return {
            "ok": True,
            "seconds": round(time.monotonic() - started, 3),
            "finish_reason": result["choices"][0]["finish_reason"],
            "content": content,
        }
    except (KeyError, OSError, ValueError, urllib.error.HTTPError) as error:
        return {
            "ok": False,
            "seconds": round(time.monotonic() - started, 3),
            "error": str(error),
        }


def validate(mode: str, result: dict[str, object]) -> str | None:
    if not result["ok"]:
        return str(result["error"])
    if result["finish_reason"] != "stop":
        return f"finish_reason={result['finish_reason']}"
    if mode == "plain" and str(result["content"]).strip() != "OK":
        return "plain response was not exactly OK"
    if mode == "json":
        try:
            decoded = json.loads(str(result["content"]))
        except json.JSONDecodeError:
            return "structured response was not JSON"
        if set(decoded) != {"label", "confidence"}:
            return "structured response did not match the schema"
    return None


def run_round(
    url: str,
    mode: str,
    concurrency: int,
    prompt_chars: int,
    request_label: str,
    timeout_seconds: int,
) -> list[dict[str, object]]:
    payload = build_payload(mode, prompt_chars, request_label)
    barrier = threading.Barrier(concurrency)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_request, url, payload, barrier, timeout_seconds)
            for _ in range(concurrency)
        ]
        return [future.result() for future in as_completed(futures)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--mode", choices=["plain", "json"], required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prompt-chars", type=int, default=12000)
    parser.add_argument("--request-label", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    if args.concurrency < 1 or args.rounds < 1 or args.prompt_chars < 1:
        parser.error("concurrency, rounds, and prompt-chars must be positive")

    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    failures: list[str] = []
    timings: list[float] = []
    for number in range(1, args.rounds + 1):
        results = run_round(
            url,
            args.mode,
            args.concurrency,
            args.prompt_chars,
            args.request_label,
            args.timeout_seconds,
        )
        errors = [error for result in results if (error := validate(args.mode, result))]
        timings.extend(float(result["seconds"]) for result in results)
        if errors:
            failures.extend(f"round {number}: {error}" for error in errors)

    summary = {
        "mode": args.mode,
        "concurrency": args.concurrency,
        "rounds": args.rounds,
        "requests": args.concurrency * args.rounds,
        "failures": failures,
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "mean_seconds": round(sum(timings) / len(timings), 3),
    }
    print(json.dumps(summary, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
