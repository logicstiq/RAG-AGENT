"""Generation backend. Picks whichever API key is set — OpenAI, Anthropic,
or Gemini, checked in that order — otherwise runs 'offline': no generation
call at all, just the retrieved passages, formatted as an answer. That
keeps the whole project runnable and demo-able with zero cost and zero
setup, which matters for a portfolio piece someone else will actually try.

Gemini is called through Google's own OpenAI-compatible endpoint (using the
openai Python library pointed at Google's base_url, not at OpenAI) rather
than the separate google-genai SDK — one less dependency, and the request/
response shape is identical to the OpenAI path above it. A Gemini key will
NOT work if it's put in OPENAI_API_KEY instead: that env var always talks
to api.openai.com, and Google's key would just be rejected there as
invalid. Use GEMINI_API_KEY specifically.
"""
from __future__ import annotations

import os

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = (
    "You are a precise retrieval-augmented assistant. Answer ONLY using the "
    "provided context passages. If the answer isn't in the context, say so "
    "plainly instead of guessing. Cite sources inline like [source_file]."
)


def _format_context(passages: list[dict]) -> str:
    blocks = []
    for p in passages:
        blocks.append(f"[{p['source']}]\n{p['text']}")
    return "\n\n---\n\n".join(blocks)


def _offline_answer(query: str, passages: list[dict]) -> str:
    if not passages:
        return "No matching passages were found in the ingested documents."
    lines = [
        "(Offline mode — no LLM API key set, showing the best-matching passages "
        "instead of a generated answer. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or "
        "GEMINI_API_KEY to get a written answer.)\n"
    ]
    for p in passages[:4]:
        snippet = p["text"][:400].strip()
        lines.append(f"From [{p['source']}] (score {p['score']:.2f}):\n{snippet}\n")
    return "\n".join(lines)


def _openai_answer(query: str, passages: list[dict]) -> str:
    from openai import OpenAI

    client = OpenAI()
    context = _format_context(passages)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def _anthropic_answer(query: str, passages: list[dict]) -> str:
    import anthropic

    client = anthropic.Anthropic()
    context = _format_context(passages)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _gemini_answer(query: str, passages: list[dict]) -> str:
    # Deliberately reuses the openai library rather than adding a
    # google-genai dependency — Google's OpenAI-compatible endpoint accepts
    # the exact same request/response shape as the real OpenAI path above.
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_BASE_URL)
    context = _format_context(passages)
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def generate_answer(query: str, passages: list[dict]) -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return _openai_answer(query, passages)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _anthropic_answer(query, passages)
    if os.environ.get("GEMINI_API_KEY"):
        return _gemini_answer(query, passages)
    return _offline_answer(query, passages)


TABLE_SYSTEM_PROMPT = (
    "You are summarizing the EXACT result of a spreadsheet filter — the "
    "numbers below are already correct and final. Do not recompute, round, "
    "or second-guess them. Just state what was found in one or two plain "
    "sentences, then show the raw result as-is underneath."
)


def _table_via_openai_compatible(query: str, filtered_text: str, api_key: str, base_url: str | None, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TABLE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nExact filter result:\n{filtered_text}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def summarize_table_result(query: str, filtered_text: str) -> str:
    """Wraps an exact pandas filter result in a short natural-language
    sentence. Never asked to produce the numbers itself — those already
    came from an exact filter in structured_query.py."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if not (openai_key or anthropic_key or gemini_key):
        return filtered_text  # offline: just show the exact result, no wrapper needed

    try:
        if openai_key:
            return _table_via_openai_compatible(query, filtered_text, openai_key, None, "gpt-4o-mini")
        if anthropic_key:
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=600,
                system=TABLE_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"Question: {query}\n\nExact filter result:\n{filtered_text}"}
                ],
            )
            return "".join(block.text for block in response.content if block.type == "text")
        return _table_via_openai_compatible(query, filtered_text, gemini_key, GEMINI_BASE_URL, GEMINI_MODEL)
    except Exception:
        # If the API call fails for any reason, the exact result is still correct —
        # never lose it behind a generation error.
        return filtered_text
