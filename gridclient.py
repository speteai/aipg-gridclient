"""
gridclient — a tiny, dependency-free Python client for the AI Power Grid API,
with the integration gotchas handled for you.

Why this exists (things that bite raw callers of the grid):
  • Reasoning models (qwen3:*, gpt-oss-120b) spend the token budget on hidden
    reasoning; at a low max_tokens you get content=None / finish_reason="length".
    -> chat() auto-raises max_tokens to a safe floor for those models.
  • A request with the default Python-urllib User-Agent is blocked with 403.
    -> every request sends an explicit User-Agent.
  • Image models are NOT in /v1/models and must be named explicitly ("auto" is
    rejected). -> image_models() discovers them; generate_image() requires a model
    and validates size (512-1536).
  • 402/413/401 come back as terse JSON. -> mapped to clear GridError messages.
  • There is no balance endpoint; account() returns what the grid does expose.

Stdlib only (Python 3.8+). No pip install.

Quick start:
    from gridclient import GridClient
    g = GridClient("grid_your_key")
    print(g.chat("qwen3:8b", "What is 17*23?"))        # -> "391" (no empty reply)
    print(g.models())                                   # text/vision models
    print(g.image_models())                             # image models
    url = g.generate_image("a red fox", model="z-image-turbo")[0]["url"]
"""
import json
import urllib.request
import urllib.error

DEFAULT_BASE = "https://api.aipowergrid.io/v1"
DEFAULT_UA = "aipg-gridclient/1.0"

# Models that spend budget on hidden reasoning and need output headroom.
REASONING_HINTS = ("qwen3", "gpt-oss")
# Safe floor for reasoning models so the visible answer survives.
REASONING_MIN_TOKENS = 1024


class GridError(Exception):
    """A clear, actionable error from the grid."""


def _looks_reasoning(model: str) -> bool:
    m = (model or "").lower()
    return any(h in m for h in REASONING_HINTS)


class GridClient:
    def __init__(self, api_key, base_url=DEFAULT_BASE, user_agent=DEFAULT_UA, timeout=120):
        if not api_key:
            raise GridError("An API key is required (get one from the AIPG dashboard).")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout = timeout

    # ---- low-level HTTP -------------------------------------------------
    def _request(self, method, path, payload=None, auth=True):
        url = self.base_url + path
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("User-Agent", self.user_agent)   # or you get a blank 403
        req.add_header("Content-Type", "application/json")
        if auth:
            req.add_header("Authorization", "Bearer " + self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            self._raise_friendly(e)
        except urllib.error.URLError as e:
            raise GridError(f"Could not reach {url}: {e.reason}") from e

    def _raise_friendly(self, e):
        try:
            body = json.loads(e.read().decode())
            detail = body.get("detail") or body.get("error") or body
        except Exception:
            detail = ""
        code = e.code
        if code == 401:
            raise GridError("401: invalid or missing API key.") from e
        if code == 402:
            raise GridError(
                "402: out of credits. The grid has no balance endpoint — "
                "top up in the dashboard; you only learn you're empty here."
            ) from e
        if code == 413:
            raise GridError(f"413: request too large (grid max is 200000 chars). {detail}") from e
        if code == 404 and isinstance(detail, str) and "not available" in detail:
            raise GridError(f"404: {detail}") from e
        raise GridError(f"{code}: {detail}") from e

    # ---- models ---------------------------------------------------------
    def models(self):
        """Text/vision model ids from the public /v1/models."""
        return [m["id"] for m in self._request("GET", "/models", auth=False).get("data", [])]

    def models_detailed(self):
        """Full model objects (modalities, pricing)."""
        return self._request("GET", "/models", auth=False).get("data", [])

    def image_models(self):
        """Image models — not in /v1/models; discovered from the API's own error."""
        try:
            self._request("POST", "/images/generations", {"model": "__discover__", "prompt": "x"})
        except GridError as e:
            msg = str(e)
            start, end = msg.find("["), msg.find("]")
            if start != -1 and end != -1:
                return [s.strip().strip("'\"") for s in msg[start + 1:end].split(",") if s.strip()]
        return []

    def account(self):
        """Linked account info the grid exposes (no credit balance available)."""
        return self._request("GET", "/account")

    # ---- chat -----------------------------------------------------------
    def chat(self, model, messages, max_tokens=None, return_raw=False, **kwargs):
        """
        Chat completion with reasoning-safe defaults.

        `messages` may be a plain string (treated as a single user turn) or a list
        of OpenAI-style message dicts. For reasoning models, max_tokens is raised
        to a safe floor so the visible answer isn't eaten by hidden reasoning.
        Returns the assistant text, or the raw response dict if return_raw=True.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if max_tokens is None:
            max_tokens = REASONING_MIN_TOKENS if _looks_reasoning(model) else 512
        elif _looks_reasoning(model) and max_tokens < REASONING_MIN_TOKENS:
            # Too small for a reasoning model — raise it rather than return empty.
            max_tokens = REASONING_MIN_TOKENS

        payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
        payload.update(kwargs)
        resp = self._request("POST", "/chat/completions", payload)
        if return_raw:
            return resp
        choice = (resp.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
        if content is None and choice.get("finish_reason") == "length":
            raise GridError(
                f"Model '{model}' returned no content — the reasoning used the whole "
                f"{max_tokens}-token budget. Call chat(..., max_tokens=<higher>)."
            )
        return content

    # ---- images ---------------------------------------------------------
    def generate_image(self, prompt, model="z-image-turbo", size="1024x1024", n=1,
                       response_format="url", **kwargs):
        """
        Generate image(s). `model` is required (the grid rejects "auto" for images).
        size width/height must be 512-1536. Returns the response `data` list, each
        item with `url` (or `b64_json`) and `seed`.
        """
        if not model:
            raise GridError("generate_image needs an explicit model (e.g. 'z-image-turbo'); "
                            "'auto' is not accepted for images. Try image_models().")
        try:
            w, h = (int(x) for x in str(size).lower().split("x"))
            if not (512 <= w <= 1536 and 512 <= h <= 1536):
                raise ValueError
        except ValueError:
            raise GridError(f"size '{size}' invalid: width and height must each be 512-1536.")
        payload = {"model": model, "prompt": prompt, "size": size, "n": n,
                   "response_format": response_format}
        payload.update(kwargs)
        return self._request("POST", "/images/generations", payload).get("data", [])


if __name__ == "__main__":
    import os, sys
    key = os.environ.get("AIPG_API_KEY")
    if not key:
        print("Set AIPG_API_KEY to try the demo.", file=sys.stderr); sys.exit(1)
    g = GridClient(key)
    print("text models :", g.models())
    print("image models:", g.image_models())
    q = sys.argv[1] if len(sys.argv) > 1 else "What is 17*23? Reply with just the number."
    print("qwen3:8b    :", g.chat("qwen3:8b", q))
