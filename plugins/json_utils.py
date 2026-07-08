"""
Shared JSON extraction utilities for LLM output.

Extracted from LocalLLMProvider (which used these as instance methods with
no actual instance state) so CloudLLMProvider can reuse the exact same
robust parsing behavior rather than duplicating or drifting from it.
"""

import json
import re


def extract_json(text: str) -> str:
    """
    Robustly extract the first syntactically complete JSON object from
    LLM output.

    Strategy:
      1. Try raw_decode on each '{' for strict JSON.
      2. If nothing parses, attempt repair_json to fix common LLM
         mistakes (unquoted keys, single-quoted strings) then retry.
      3. Raise ValueError with the first 200 chars for debugging.
    """
    decoder = json.JSONDecoder()

    def _try_decode(s: str):
        for i, ch in enumerate(s):
            if ch == '{':
                try:
                    obj, _ = decoder.raw_decode(s, i)
                    return json.dumps(obj)
                except json.JSONDecodeError:
                    continue
        return None

    result = _try_decode(text)
    if result:
        return result

    repaired = repair_json(text)
    result = _try_decode(repaired)
    if result:
        print("[LLM] JSON repaired (unquoted keys or single quotes fixed)", flush=True)
        return result

    raise ValueError(
        f"No valid JSON object found in LLM output. "
        f"First 200 chars: {text[:200]!r}"
    )


def repair_json(text: str) -> str:
    """
    Best-effort fix for common LLM JSON mistakes:
    - Unquoted object keys:  { key: value }  ->  { "key": value }
    - Single-quoted strings: { 'k': 'v' }   ->  { "k": "v" }
    """
    repaired = re.sub(r'(?<!["\w])([a-zA-Z_]\w*)\s*:', r'"\1":', text)
    repaired = re.sub(r"'([^']*)'", r'"\1"', repaired)
    return repaired
