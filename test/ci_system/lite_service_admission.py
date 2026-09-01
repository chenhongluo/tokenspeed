#!/usr/bin/env python3
"""Bounded OpenAI-compatible admission client for the Lite 8P8D service."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass(frozen=True)
class CompletionSummary:
    case: str
    submitted_s: float
    first_token_s: float
    completed_s: float
    http_status: int
    completion_tokens: int
    finish_reason: str
    output_sha256: str


class AdmissionError(RuntimeError):
    pass


def _digest(text: str, completion_tokens: int) -> str:
    payload = json.dumps(
        {"completion_tokens": completion_tokens, "text": text},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _request(
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None,
    timeout: float,
):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    return _OPENER.open(request, timeout=timeout)


def _payload(model: str, prompt: str, max_tokens: int, *, stream: bool) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _completion_summary(
    *,
    case: str,
    origin: float,
    submitted: float,
    first_token: float,
    completed: float,
    status: int,
    text: str,
    completion_tokens: int,
    finish_reason: str | None,
    expected_tokens: int,
) -> CompletionSummary:
    if status != 200:
        raise AdmissionError(f"{case}: HTTP status is {status}, expected 200")
    if completion_tokens != expected_tokens:
        raise AdmissionError(
            f"{case}: completion_tokens={completion_tokens}, expected {expected_tokens}"
        )
    if not text:
        raise AdmissionError(f"{case}: completion text is empty")
    if not finish_reason:
        raise AdmissionError(f"{case}: finish_reason is missing")
    return CompletionSummary(
        case=case,
        submitted_s=submitted - origin,
        first_token_s=first_token - origin,
        completed_s=completed - origin,
        http_status=status,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        output_sha256=_digest(text, completion_tokens),
    )


def request_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    case: str,
    origin: float,
    timeout: float,
) -> CompletionSummary:
    submitted = time.monotonic()
    with _request(
        base_url,
        "/v1/completions",
        payload=_payload(model, prompt, max_tokens, stream=False),
        timeout=timeout,
    ) as response:
        status = response.getcode()
        body = json.load(response)
    completed = time.monotonic()
    choices = body.get("choices") or []
    if len(choices) != 1:
        raise AdmissionError(f"{case}: expected one completion choice")
    usage = body.get("usage") or {}
    return _completion_summary(
        case=case,
        origin=origin,
        submitted=submitted,
        first_token=completed,
        completed=completed,
        status=status,
        text=choices[0].get("text") or "",
        completion_tokens=int(usage.get("completion_tokens", 0)),
        finish_reason=choices[0].get("finish_reason"),
        expected_tokens=max_tokens,
    )


def stream_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    case: str,
    origin: float,
    timeout: float,
    on_first_token: Callable[[float], None] | None = None,
) -> CompletionSummary:
    submitted = time.monotonic()
    pieces: list[str] = []
    completion_tokens = 0
    finish_reason = None
    first_token = None
    with _request(
        base_url,
        "/v1/completions",
        payload=_payload(model, prompt, max_tokens, stream=True),
        timeout=timeout,
    ) as response:
        status = response.getcode()
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line.removeprefix("data:").strip()
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            usage = chunk.get("usage") or {}
            if "completion_tokens" in usage:
                completion_tokens = int(usage["completion_tokens"])
            choices = chunk.get("choices") or []
            if choices and first_token is None:
                first_token = time.monotonic()
                if on_first_token is not None:
                    on_first_token(first_token)
            for choice in choices:
                pieces.append(choice.get("text") or "")
                finish_reason = choice.get("finish_reason") or finish_reason
    completed = time.monotonic()
    if first_token is None:
        raise AdmissionError(f"{case}: stream returned no token event")
    return _completion_summary(
        case=case,
        origin=origin,
        submitted=submitted,
        first_token=first_token,
        completed=completed,
        status=status,
        text="".join(pieces),
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        expected_tokens=max_tokens,
    )


def _model_ids(base_url: str, timeout: float) -> list[str]:
    with _request(base_url, "/v1/models", payload=None, timeout=timeout) as response:
        if response.getcode() != 200:
            raise AdmissionError(f"model list returned HTTP {response.getcode()}")
        body = json.load(response)
    return [str(item["id"]) for item in body.get("data") or [] if "id" in item]


def run_admission(
    *,
    base_url: str,
    model: str,
    prompts: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    if set(prompts) != {"A", "B"} or any(not value for value in prompts.values()):
        raise AdmissionError("prompt file must contain non-empty A and B strings")
    models = _model_ids(base_url, timeout)
    if models.count(model) != 1:
        raise AdmissionError(f"served model {model!r} must appear exactly once")

    origin = time.monotonic()
    summaries = [
        request_completion(
            base_url,
            model,
            prompts["A"],
            4,
            case=case,
            origin=origin,
            timeout=timeout,
        )
        for case in ("fresh-a-1", "fresh-a-2", "slot-reuse-a")
    ]
    if len({item.output_sha256 for item in summaries}) != 1:
        raise AdmissionError("fresh/slot-reuse A digests differ")

    standalone_b = request_completion(
        base_url,
        model,
        prompts["B"],
        4,
        case="standalone-b",
        origin=origin,
        timeout=timeout,
    )
    summaries.append(standalone_b)

    first_a = threading.Event()
    stream_result: list[CompletionSummary] = []
    stream_error: list[BaseException] = []

    def run_a() -> None:
        try:
            stream_result.append(
                stream_completion(
                    base_url,
                    model,
                    prompts["A"],
                    32,
                    case="late-a-stream",
                    origin=origin,
                    timeout=timeout,
                    on_first_token=lambda _timestamp: first_a.set(),
                )
            )
        except BaseException as exc:  # preserve the worker's original failure
            stream_error.append(exc)
            first_a.set()

    worker = threading.Thread(target=run_a, name="lite-late-a", daemon=True)
    worker.start()
    if not first_a.wait(timeout):
        raise AdmissionError("late A did not produce a first token before timeout")
    if stream_error:
        raise stream_error[0]
    late_b = request_completion(
        base_url,
        model,
        prompts["B"],
        4,
        case="late-b",
        origin=origin,
        timeout=timeout,
    )
    worker.join(timeout)
    if worker.is_alive():
        raise AdmissionError("late A stream did not complete before timeout")
    if stream_error:
        raise stream_error[0]
    late_a = stream_result[0]
    if not late_a.first_token_s < late_b.submitted_s < late_a.completed_s:
        raise AdmissionError(
            "B was not submitted after A's first token while A was live"
        )
    if late_b.output_sha256 != standalone_b.output_sha256:
        raise AdmissionError("late-admission B differs from standalone B")
    summaries.extend((late_a, late_b))

    replay_b = request_completion(
        base_url,
        model,
        prompts["B"],
        4,
        case="post-bs2-b",
        origin=origin,
        timeout=timeout,
    )
    if replay_b.output_sha256 != standalone_b.output_sha256:
        raise AdmissionError("post-BS2 B differs from standalone B")
    summaries.append(replay_b)
    return {
        "schema_version": 1,
        "passed": True,
        "served_model": model,
        "model_list_count": len(models),
        "cases": [asdict(summary) for summary in summaries],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prompts = json.loads(args.prompt_file.read_text())
    report = run_admission(
        base_url=args.base_url,
        model=args.model,
        prompts=prompts,
        timeout=args.timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
