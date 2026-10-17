"""Single-page local web UI (standard library only). Binds 127.0.0.1 by default.

On EC2, view it through SSM port forwarding instead of opening a port:
    aws ssm start-session --target <instance-id> \
      --document-name AWS-StartPortForwardingSession --parameters portNumber=8080,localPortNumber=8080
"""

from __future__ import annotations

import html
import tempfile
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .triage import Deps, triage

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Apply-or-Skip</title>
<style>body{{font-family:system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem}}
textarea{{width:100%;height:120px}}table{{border-collapse:collapse;width:100%;margin-top:1rem}}
td,th{{border:1px solid #ddd;padding:.35rem .5rem;font-size:.9rem;vertical-align:top}}
.bar{{background:#4f8cff;height:.6rem;display:inline-block}}.src-decider{{color:#0a7}}.src-llm_fallback{{color:#c60}}
.src-rule,.src-low_confidence{{color:#b00}}pre{{white-space:pre-wrap;background:#f6f6f6;padding:1rem}}
.banner{{background:#fee;border:1px solid #f99;padding:.5rem}}</style></head><body>
<h1>Apply-or-Skip <small>#strands-decider-2b #tool-selection</small></h1>{banner}
<form method="post"><p>Job URL, saved posting path, or paste the posting text:</p>
<textarea name="source">{source}</textarea><p><button>Triage</button></p></form>{result}</body></html>"""


def render_result(st) -> str:
    rows = []
    for r in st.steps:
        bar = int(r.get("p_top", 0) * 120)
        rows.append(
            f"<tr><td>{r['step']}</td><td><b>{html.escape(r['tool'])}</b></td>"
            f"<td><span class='bar' style='width:{bar}px'></span> {r.get('p_top', 0):.2f}</td>"
            f"<td>{r.get('latency_ms', 0):.0f} ms</td><td class='src-{r['source']}'>{r['source']}</td>"
            f"<td>{html.escape(r['result'][:220])}</td></tr>")
    return ("<h2>Decision trace</h2><table><tr><th>#</th><th>tool</th><th>p(top)</th><th>decider</th>"
            "<th>decided by</th><th>result</th></tr>" + "".join(rows) + "</table>"
            f"<h2>Outcome: {html.escape(str(st.outcome))}</h2><pre>{html.escape(st.output)}</pre>")


def make_handler(deps: Deps, profile: dict[str, Any] | None):
    mock = deps.settings.decider_backend == "mock"
    banner = "<p class='banner'>OFFLINE MOCK MODE: keyword heuristics, not Decider 2B.</p>" if mock else ""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: str, code: int = 200) -> None:
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            self._send(PAGE.format(banner=banner, source="", result=""))

        def do_POST(self) -> None:  # noqa: N802
            length = min(int(self.headers.get("Content-Length", 0)), 200_000)
            form = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))
            source = (form.get("source") or [""])[0].strip()
            target = source
            if source and "\n" in source:  # pasted text -> temp file
                with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as fh:
                    fh.write(source)
                    target = fh.name
            st = triage(target, profile, deps)
            self._send(PAGE.format(banner=banner, source=html.escape(source[:5000]), result=render_result(st)))

        def log_message(self, *a: Any) -> None:  # quiet
            pass

    return Handler


def serve(deps: Deps, profile: dict[str, Any] | None, host: str = "127.0.0.1", port: int = 8080) -> None:
    srv = ThreadingHTTPServer((host, port), make_handler(deps, profile))
    print(f"Apply-or-Skip UI on http://{host}:{port}")
    srv.serve_forever()
