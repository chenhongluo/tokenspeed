import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from test.ci_system.lite_service_admission import AdmissionError, run_admission

import pytest


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        pass

    def _json(self, body):
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        assert self.path == "/v1/models"
        self._json({"data": [{"id": "lite"}]})

    def do_POST(self):
        assert self.path == "/v1/completions"
        size = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(size))
        marker = "A" if body["prompt"] == "alpha" else "B"
        count = int(body["max_tokens"])
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for index in range(count):
                chunk = {
                    "choices": [
                        {
                            "text": marker,
                            "finish_reason": "length" if index == count - 1 else None,
                        }
                    ]
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.005)
            usage = {"usage": {"completion_tokens": count}, "choices": []}
            self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self._json(
            {
                "choices": [
                    {"text": marker * count, "finish_reason": "length", "index": 0}
                ],
                "usage": {"completion_tokens": count},
            }
        )


@pytest.fixture
def service():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_full_admission_matrix_is_anonymous_and_late_admits(service):
    report = run_admission(
        base_url=service,
        model="lite",
        prompts={"A": "alpha", "B": "beta"},
        timeout=5,
    )

    assert report["passed"] is True
    assert [case["case"] for case in report["cases"]] == [
        "fresh-a-1",
        "fresh-a-2",
        "slot-reuse-a",
        "standalone-b",
        "late-a-stream",
        "late-b",
        "post-bs2-b",
    ]
    assert "alpha" not in json.dumps(report)
    assert "beta" not in json.dumps(report)
    cases = {case["case"]: case for case in report["cases"]}
    assert (
        cases["late-a-stream"]["first_token_s"]
        < cases["late-b"]["submitted_s"]
        < cases["late-a-stream"]["completed_s"]
    )
    assert cases["standalone-b"]["output_sha256"] == cases["late-b"]["output_sha256"]


def test_admission_rejects_bad_prompt_contract(service):
    with pytest.raises(AdmissionError, match="non-empty A and B"):
        run_admission(
            base_url=service,
            model="lite",
            prompts={"A": "alpha"},
            timeout=5,
        )
