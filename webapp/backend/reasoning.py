"""
Plain-language reasoning module: turns a classification result into a short,
non-technical written explanation via a hosted, OpenAI-compatible chat
completion API.

Provider-agnostic on purpose -- any OpenAI-compatible endpoint (Groq, NVIDIA
NIM, OpenRouter, a local vLLM server, ...) works by setting three environment
variables. Configured against NVIDIA's hosted API in webapp/backend/.env at
the time this was built; swapping to Groq later is a two-line env change,
no code change.
"""

import os

from openai import OpenAI

_client: OpenAI | None = None
_client_checked = False


def _get_client() -> OpenAI | None:
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True

    api_key = os.environ.get("REASONING_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("REASONING_BASE_URL", "https://integrate.api.nvidia.com/v1")
    _client = OpenAI(base_url=base_url, api_key=api_key)
    return _client


_MODEL_NAME = os.environ.get("REASONING_MODEL", "meta/llama-3.1-8b-instruct")

_DOMAIN_CONTEXT = {
    "cifar100": (
        "an everyday photograph, being sorted into one of a hundred common "
        "object, animal, or scene categories"
    ),
    "busi": (
        "a breast ultrasound scan, being sorted into one of three diagnostic "
        "categories (benign, malignant, normal) -- for demonstration purposes "
        "only, not a medical judgement"
    ),
}


def explain(domain: str, predicted_label: str, confidence: float, top_predictions: list[dict]) -> str:
    """Return a short, human-readable explanation of the model's decision.

    Falls back to a clear placeholder message (not an exception) if no
    reasoning-engine API key is configured, so the rest of the demo keeps
    working without it.
    """
    client = _get_client()
    if client is None:
        return (
            "Reasoning module not configured — set REASONING_API_KEY (and, if needed, "
            "REASONING_BASE_URL / REASONING_MODEL) in webapp/backend/.env to enable "
            "plain-language explanations. The classification and visual explanation "
            "above are unaffected."
        )

    alternatives = ", ".join(
        f"{p['label']} ({p['confidence'] * 100:.1f}%)" for p in top_predictions[1:4]
    )
    prompt = (
        f"An image-classification system just analysed {_DOMAIN_CONTEXT.get(domain, 'an image')}. "
        f"It decided the image most likely shows: '{predicted_label}', with "
        f"{confidence * 100:.1f}% confidence. Its next most likely alternatives, in order, "
        f"were: {alternatives}. The system also produced a heatmap highlighting the region "
        "of the image that most influenced this decision. In two or three short, plain-English "
        "sentences, explain this result to someone with no machine-learning background, as if "
        "you were a colleague talking them through it. Do not use technical jargon (no mentions "
        "of neurons, layers, gradients, tensors, softmax, or attention). Be concrete and "
        "confident, but honest that this is a probabilistic best guess, not a certainty."
    )

    try:
        response = client.chat.completions.create(
            model=_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=220,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001 -- surfaced to the UI as-is, not swallowed
        return f"Reasoning module error: {exc}"
