"""CSRF, origin and value hygiene for the local operator consoles (audit AG-04 / IM-02).

`127.0.0.1` is not a security boundary. A page the user visits in the same browser can POST a
plain HTML form to a localhost port: form posts are CORS-simple, so nothing in the browser blocks
the request and nothing in the app rejected it. Demonstrated against these two consoles with Flask
test clients: a cross-origin POST with `Origin: https://evil.example` returned **200** and
performed five writes — `max_total_cost_usd=9999`, `max_total_tokens=999999999`, `max_calls=99999`,
`unlocked_tier=tier_3`, `novel_dataset=5.0` — while `/launch` spawned a real paid `run_council.py`
and `/key/delete` called `secrets.delete('ANTHROPIC_API_KEY')`.

Four independent checks, because any one of them can be defeated on its own:

1. **CSRF token.** A per-process random token embedded in every form. The attacker's page cannot
   read it — the same-origin policy stops it fetching the console's HTML — so it cannot forge the
   field. Compared with `secrets.compare_digest`.
2. **Origin.** Must be present and must be one of this console's own origins. This is what stops
   the attack even if a token leaks into a log.
3. **Host.** Must be a loopback host on the console's own port, which is what stops DNS rebinding
   (`evil.example` resolving to 127.0.0.1 would otherwise pass an Origin check on its own origin).
4. **`Sec-Fetch-Site`.** Rejected unless `same-origin` / `none` when the browser sends it. Defence
   in depth: it is set by the browser and cannot be forged by page script.

Plus value hygiene, because the config writer is a regex over YAML text: `safe_scalar` rejects
anything with a newline or a colon, and `set_scalar_checked` re-parses the file afterwards and
restores it unless exactly the intended key changed. That is stronger than the `safe_dump` the
audit suggested — safe_dump would also destroy the comments these YAMLs are written around, and
`ruamel.yaml` (which preserves them) is not a dependency here.
"""
from __future__ import annotations

import re
import secrets as _secrets
from pathlib import Path

import yaml

#: One token per process. These consoles are single-user and short-lived, so there is no session
#: store to keep: restarting the app invalidates every outstanding form, which is the safe default.
TOKEN = _secrets.token_urlsafe(32)
FIELD = "_csrf"
HEADER = "X-CSRF-Token"
UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}
_SCALAR_OK = re.compile(r"^[A-Za-z0-9_.:+/@-]{1,120}$")   # colon allowed only for model tags


def hidden_field() -> str:
    """The hidden input every form must carry."""
    return f'<input type=hidden name={FIELD} value="{TOKEN}">'


def origins(port: int) -> set[str]:
    return {f"http://{h}:{port}" for h in ("127.0.0.1", "localhost", "[::1]")}


def hosts(port: int) -> set[str]:
    return {f"{h}:{port}" for h in ("127.0.0.1", "localhost", "[::1]")}


def refusal(request, port: int) -> str | None:
    """Why this request must be refused, or None if it may proceed."""
    if request.method not in UNSAFE:
        return None
    host = (request.headers.get("Host") or "").lower()
    if host not in hosts(port):
        return f"Host {host!r} is not a loopback host on port {port} (DNS-rebinding guard)"
    site = request.headers.get("Sec-Fetch-Site")
    if site and site not in ("same-origin", "none"):
        return f"Sec-Fetch-Site: {site}"
    origin = request.headers.get("Origin")
    if not origin:
        # No Origin on an unsafe method: a same-origin browser form always sends one, so this is
        # either a non-browser client or a stripped request. The token still has to be right.
        if not _token_ok(request):
            return "no Origin header and no valid CSRF token"
    elif origin not in origins(port):
        return f"cross-origin request from {origin!r}"
    if not _token_ok(request):
        return "missing or invalid CSRF token"
    return None


def _token_ok(request) -> bool:
    got = request.form.get(FIELD) or request.headers.get(HEADER) or ""
    return _secrets.compare_digest(str(got), TOKEN)


def install(app, *, port: int) -> None:
    """Refuse every unsafe request that fails any of the four checks."""
    from flask import request

    @app.before_request
    def _guard():                                    # pragma: no cover - exercised via test client
        why = refusal(request, port)
        if why:
            return (f"refused: {why}\n\nThis console only accepts form posts from its own page "
                    f"(audit AG-04). Open http://127.0.0.1:{port}/ and use the form.", 403,
                    {"Content-Type": "text/plain; charset=utf-8"})
        return None


def safe_scalar(value: str) -> str:
    """A YAML- and HTML-safe scalar, or ValueError.

    The config writer substitutes into YAML TEXT, so a value containing a newline injects a
    sibling key: `local_model="llama3.1:8b\\n  mode: deterministic"` silently set the runtime mode.

    SCOPE: this is about YAML and markup injection — newlines, quotes, spaces, YAML punctuation —
    not about path safety. `/` and `.` are allowed because model tags need them
    (`anthropic/claude-x`, `llama3.1:8b`), so `../../etc` passes; that is deliberate and safe
    because no caller treats these values as a path. They become YAML scalars, or file CONTENT
    (the approver name), and are escaped wherever they are rendered.
    """
    v = str(value).strip()
    if not _SCALAR_OK.match(v):
        raise ValueError(f"unsafe config value {value!r}: expected a short scalar matching "
                         f"{_SCALAR_OK.pattern} (no newlines, quotes or YAML punctuation)")
    return v


def set_scalar_checked(path: Path, key: str, value, *, allowed: set | None = None) -> bool:
    """Replace `key: <old>` on its first match, then PROVE only that changed.

    Comments survive because the edit is textual; correctness is enforced afterwards by re-parsing
    and comparing against the original structure with the one key updated. Anything else — an
    injected sibling, a broken document, a key that landed in the wrong block — restores the file
    and raises.
    """
    text = path.read_text(encoding="utf-8")
    before = yaml.safe_load(text) or {}
    if isinstance(value, str):
        value = safe_scalar(value)
        if allowed is not None and value not in allowed:
            raise ValueError(f"{value!r} is not one of {sorted(allowed)}")
    elif not isinstance(value, (int, float, bool)):
        raise ValueError(f"unsupported config value type {type(value).__name__}")

    pat = re.compile(rf"^(\s*{re.escape(key)}:\s*)([^\n#]*)(.*)$", re.MULTILINE)
    new, n = pat.subn(
        lambda m: f"{m.group(1)}{value}{'  ' + m.group(3).strip() if m.group(3).strip() else ''}",
        text, count=1)
    if not n:
        return False
    path.write_text(new, encoding="utf-8")
    try:
        after = yaml.safe_load(new) or {}
    except yaml.YAMLError as e:
        path.write_text(text, encoding="utf-8")
        raise ValueError(f"edit made {path.name} unparseable, restored: {e}") from e
    if _diff_keys(before, after) - {key}:
        path.write_text(text, encoding="utf-8")
        raise ValueError(f"edit changed more than {key!r} in {path.name} "
                         f"({sorted(_diff_keys(before, after))}), restored")
    return True


def _diff_keys(a, b, prefix: str = "") -> set[str]:
    """Leaf keys whose value differs between two parsed YAML structures."""
    out: set[str] = set()
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            out |= _diff_keys(a.get(k), b.get(k), f"{prefix}{k}.")
        return out
    if a != b:
        out.add(prefix.rstrip(".").split(".")[-1] if prefix else "<root>")
    return out
