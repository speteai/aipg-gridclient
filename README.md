# gridclient

A tiny, **dependency-free** Python client for the
[AI Power Grid](https://api.aipowergrid.io/v1) API — with the integration
gotchas handled for you. One file, stdlib only, Python 3.8+.

> Unofficial, community-built. Not affiliated with AI Power Grid.

## Why

The grid is OpenAI-compatible, so you *can* use the official `openai` SDK
(see [below](#using-the-official-openai-sdk)). But a few sharp edges bite raw
callers. `gridclient` smooths them:

| Gotcha | What `gridclient` does |
|---|---|
| Reasoning models (`qwen3:*`, `gpt-oss-120b`) return `content=None` / `finish_reason:"length"` when `max_tokens` is too small — the hidden reasoning eats the budget. | `chat()` auto-raises `max_tokens` to a safe floor (1024) for those models, and raises a clear error if a reply is *still* empty. |
| A request with the default `Python-urllib` User-Agent is blocked with a blank **403**. | Every request sends an explicit `User-Agent`. |
| Image models are **not** in `/v1/models`, and `"auto"` is rejected for images. | `image_models()` discovers them; `generate_image()` requires a model and validates `size` (512–1536). |
| `402` / `413` / `401` come back as terse JSON. | Mapped to clear, actionable `GridError` messages. |
| No balance endpoint exists. | `account()` returns what the grid does expose; the docstring says so. |

## Install

Just copy `gridclient.py` into your project. No dependencies.

```bash
curl -O https://raw.githubusercontent.com/speteai/aipg-gridclient/main/gridclient.py
```

## Use

```python
from gridclient import GridClient, GridError

g = GridClient("grid_your_key")          # or GridClient(os.environ["AIPG_API_KEY"])

# Chat — reasoning-safe. This returns "391", not an empty string:
print(g.chat("qwen3:8b", "What is 17*23? Reply with just the number."))

# Pass messages and params through like the OpenAI API:
print(g.chat(
    "deepseek-v4-flash-nvfp4",
    [{"role": "system", "content": "You are terse."},
     {"role": "user", "content": "One word: capital of France?"}],
    max_tokens=10, temperature=0,
))

# Discover models
print(g.models())          # ['auto', 'deepseek-v4-flash-nvfp4', 'gpt-oss-120b', 'qwen3:8b']
print(g.image_models())    # ['FLUX.2 Klein 4B FP8', 'Krea 2 Turbo', 'MOSS Image SFT BF16', 'z-image-turbo']

# Vision (deepseek accepts image input) — standard OpenAI content blocks:
g.chat("deepseek-v4-flash-nvfp4", [{"role": "user", "content": [
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "https://…"}},
]}])

# Image generation — model is required ("auto" is rejected for images):
imgs = g.generate_image("a red fox in snow", model="z-image-turbo", size="1024x1024")
print(imgs[0]["url"], imgs[0]["seed"])

# Clear errors
try:
    g.chat("nope", "hi")
except GridError as e:
    print(e)   # 404: Model 'nope' is not available. Online models: [...]
```

Run the built-in demo:

```bash
export AIPG_API_KEY=grid_your_key
python3 gridclient.py "Explain prime numbers in one sentence."
```

## Using the official OpenAI SDK

If you'd rather use the `openai` package, point it at the grid — just remember the
reasoning `max_tokens` gotcha yourself:

```python
from openai import OpenAI
client = OpenAI(base_url="https://api.aipowergrid.io/v1", api_key="grid_...")
r = client.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "What is 17*23?"}],
    max_tokens=1024,      # reasoning models need headroom or content is None
)
```

(The official SDK sets its own User-Agent, so the 403 gotcha doesn't apply there.)

## Not available on the grid

`/v1/embeddings`, `/v1/completions` (legacy), `/v1/audio/*`, `/v1/moderations`,
`/v1/rerank` all return 404 — don't assume full OpenAI parity. Chat, vision and
image generation are supported.

## License

MIT — see [LICENSE](LICENSE).
