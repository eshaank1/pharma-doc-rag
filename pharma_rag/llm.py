import json
import re
import time

import requests

from .config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_NUM_CTX


def wait_for_ollama(timeout: int = 120) -> bool:
    """Block until the Ollama server responds, or timeout. Needed because
    `depends_on` in docker-compose only waits for the container to start,
    not for the Ollama server inside it to finish booting."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5).ok:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def ensure_model_pulled() -> None:
    """Pull OLLAMA_MODEL if it isn't already present, so `docker compose up`
    works out of the box on a machine that has never run this model before."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        resp.raise_for_status()
        local_models = {m["name"] for m in resp.json().get("models", [])}
    except Exception as e:
        print(f"Could not reach Ollama to check installed models: {e}")
        return

    have_it = any(m == OLLAMA_MODEL or m.startswith(f"{OLLAMA_MODEL}:") for m in local_models)
    if have_it:
        print(f"Ollama model '{OLLAMA_MODEL}' already present.")
        return

    print(f"Pulling Ollama model '{OLLAMA_MODEL}' (first run only, can take several minutes)...")
    try:
        with requests.post(
            f"{OLLAMA_HOST}/api/pull",
            json={"name": OLLAMA_MODEL, "stream": True},
            stream=True,
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                status = data.get("status", "")
                if data.get("total"):
                    pct = data.get("completed", 0) / data["total"] * 100
                    print(f"  {status}: {pct:.1f}%", end="\r")
                else:
                    print(f"  {status}")
        print(f"\nModel '{OLLAMA_MODEL}' ready.")
    except Exception as e:
        print(f"Failed to pull model '{OLLAMA_MODEL}': {e}")


def llm_generate(prompt: str, temperature: float = 0.1, max_new_tokens: int = 512) -> str:
    """Single entry point for all LLM calls. Talks to a local Ollama server."""
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens,
                "num_ctx": OLLAMA_NUM_CTX,
            },
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def llm_self_test() -> None:
    """Mirrors the notebook's startup self-test: confirm the LLM actually
    answers before the app is live, and print an actionable error if not."""
    try:
        reply = llm_generate("Reply with the single word: OK", max_new_tokens=5)
        print(f"LLM self-test succeeded. Model replied: {reply!r}")
    except Exception as e:
        print("=" * 70)
        print(f"LLM SELF-TEST FAILED: {type(e).__name__}: {e!r}")
        print("The app will still launch, but every AI-generated answer will fail")
        print("until this is fixed. Common causes:")
        print("  1. Ollama isn't running -- start it with: ollama serve")
        print(f"  2. The model '{OLLAMA_MODEL}' isn't pulled -- run: ollama pull {OLLAMA_MODEL}")
        print(f"  3. OLLAMA_HOST is wrong -- currently: {OLLAMA_HOST}")
        print("=" * 70)


def extract_json(text: str) -> dict:
    """Robustly pull the first JSON object out of an LLM response."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}
