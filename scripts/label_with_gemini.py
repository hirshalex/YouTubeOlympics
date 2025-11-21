# scripts/gemini_label_neg.py
import os
import time
import re
import json
import pandas as pd
from collections import defaultdict, Counter
from typing import Tuple

# Gemini SDK + Pydantic for structured output
import google.genai as genai
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from google.genai.types import HarmCategory, HarmBlockThreshold

# -------------------------
# 0. Quick config
# -------------------------
# Make sure GEMINI_API_KEY is exported in your environment:
# Windows PowerShell example:
#   $env:GEMINI_API_KEY="your_key_here"
# or set it in your OS env before running Python.
MODEL_NAME = "gemini-2.5-flash"
INPUT_PARQUET = "data/processed/sampled_neg.parquet"
OUT_PREFIX = "data/processed/gemini_labeled_sampled_neg"
SAVE_EVERY = 50                # save checkpoint every N rows
RATE_LIMIT_SLEEP = 1.5         # seconds between requests
RETRIES = 3
RETRY_WAIT = 1.0

# -------------------------
# 1. Client init
# -------------------------
try:
    client = genai.Client()  # will read GEMINI_API_KEY from env
except Exception as e:
    print("Error initializing Gemini client:", e)
    print("Make sure GEMINI_API_KEY is set in the environment.")
    raise SystemExit(1)

# -------------------------
# 2. Structured output schema (Pydantic)
# -------------------------
class VideoLabel(BaseModel):
    label: Literal["olympic", "other_sport", "non_sport"] = Field(
        description="One of 'olympic', 'other_sport', or 'non_sport'"
    )
    reason: str = Field(
        description="A one-sentence explanation for the chosen label based on title/tags."
    )

# convert Pydantic JSON Schema (this is passed to the API as response_schema)
RESPONSE_SCHEMA = VideoLabel.model_json_schema()

# -------------------------
# 3. Prompt template loader
# -------------------------
with open("scripts/groq_prompt_examples.md", "r", encoding="utf-8") as f:
    base_prompt = f.read().strip()

def build_prompt(title: str, tags: str) -> str:
    prompt = base_prompt.replace("<<title>>", str(title or "").replace("\n", " "))
    prompt = prompt.replace("<<tags>>", str(tags or "").replace("\n", " "))
    return prompt

# -------------------------
# 4. Robust response extraction (SIMPLIFIED)
# -------------------------
def extract_text(resp) -> str:
    """
    Safely extracts the generated text content from the response object.
    Returns an empty string if text is None (due to block or error).
    """
    try:
        # This is the intended and correct way to get the text from the SDK.
        text_content = getattr(resp, "text", None)
        if text_content:
            return text_content.strip()
    except Exception:
        # Ignore exceptions during extraction and proceed to check candidates
        pass
    
    # If text is None, check for safety block/reason
    if resp and getattr(resp, "candidates", None):
        reason = resp.candidates[0].finish_reason.name if resp.candidates[0].finish_reason else "UNKNOWN"
        print(f"[BLOCK] Response suppressed by Gemini API. Finish Reason: {reason}")
    
    return "" # Returns empty string on failure or block

# -------------------------
# 5. Gemini call wrapper
# -------------------------
def call_gemini(prompt: str, retries: int = RETRIES, wait: float = RETRY_WAIT) -> str:
    """
    Calls the Gemini API configured to return JSON (response_schema).
    Returns the raw string extracted from the response (may contain JSON or other text).
    """
    for attempt in range(1, retries + 1):
        try:
            # Use plain dicts for safety_settings to avoid SDK enum/constructor mismatch
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "max_output_tokens": 512,
                    "response_mime_type": "application/json",
                    "response_schema": RESPONSE_SCHEMA,
                    # pass safety settings as plain JSON-compatible dicts
                    "safety_settings": [
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_ONLY_HIGH"
                        },
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_ONLY_HIGH"
                        }
                    ],
                },
            )
            raw_text = extract_text(resp)
            return raw_text
        except Exception as e:
            print(f"[WARN] Gemini call failed (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(wait * (2 ** (attempt - 1)))
            else:
                print("[ERROR] Max retries reached; returning empty string.")
                return ""
    return ""

# -------------------------
# 6. Parse JSON / fallback
# -------------------------
def _balance_braces_and_quotes(s: str) -> str:
    """Try small repairs: balance braces and close dangling quotes."""
    # Balance braces
    opens = s.count("{")
    closes = s.count("}")
    if opens > closes:
        s = s + "}" * (opens - closes)

    # If odd number of double-quotes, append a closing quote
    dq = s.count('"')
    if dq % 2 == 1:
        s = s + '"'

    # Final attempt: if it ends with something like ... and (no closing quote),
    # strip incomplete trailing words after the last closing quote if present
    if '"' in s:
        last_quote_ix = s.rfind('"')
        # if last_quote_ix is not at final quote before } then try to trim trailing partial token
        if not s.strip().endswith('}'):
            # try to append } if makes sense
            if s.count("{") == s.count("}"):
                s = s + "}"
    return s

def parse_json_response(raw: str) -> Tuple[str, str, str]:
    """
    Parse raw model output into (label, reason, raw).
    - Tries strict pydantic validation first (VideoLabel.model_validate_json).
    - If that fails, attempts small repairs to JSON and retries.
    - If still failing, falls back to regex to extract label and a truncated reason.
    Returns (label, reason, raw_text).
    """
    if not raw or not raw.strip():
        return "non_sport", "empty_response", raw

    raw = raw.strip()

    # 1) Try strict validation with pydantic (fast path)
    try:
        parsed = VideoLabel.model_validate_json(raw)
        label = parsed.label.lower()
        reason = parsed.reason
        if label not in {"olympic", "other_sport", "non_sport"}:
            label = "non_sport"
        return label, reason, raw
    except Exception as e:
        # keep the error around for debug logging
        strict_err = e

    # 2) Try to extract first {...} block then validate
    m = re.search(r"\{.*\}", raw, flags=re.S)
    candidate = m.group(0) if m else raw

    # try direct json.loads first (faster than pydantic for broken JSON)
    try:
        obj = json.loads(candidate)
        # if loads succeeded, validate with pydantic
        try:
            parsed = VideoLabel.model_validate(obj)
            label = parsed.label.lower()
            reason = parsed.reason
            if label not in {"olympic", "other_sport", "non_sport"}:
                label = "non_sport"
            return label, reason, raw
        except Exception:
            # pydantic failed for some reason; fallback to dict extraction below
            pass
    except Exception:
        pass

    # 3) Attempt small repairs (balance braces/quotes) then parse
    repaired = _balance_braces_and_quotes(candidate)
    try:
        obj = json.loads(repaired)
        try:
            parsed = VideoLabel.model_validate(obj)
            label = parsed.label.lower()
            reason = parsed.reason
            if label not in {"olympic", "other_sport", "non_sport"}:
                label = "non_sport"
            return label, reason, raw
        except Exception:
            # still not valid pydantic; fall through to regex fallback
            pass
    except Exception:
        # json.loads failed on repaired text; fall through
        pass

    # 4) Last-resort: regex extraction of label and a truncated reason
    # extract label (strict)
    lab_m = re.search(r'"label"\s*:\s*"([^"]+)"', candidate)
    label = lab_m.group(1).lower() if lab_m else "non_sport"
    if label not in {"olympic", "other_sport", "non_sport"}:
        label = "non_sport"

    # extract reason (may be truncated) — capture up to next quote or end
    reason_m = re.search(r'"reason"\s*:\s*"([^"]*)', candidate)
    if reason_m:
        reason = reason_m.group(1).strip()
        # if truncated and ends with an incomplete word, trim trailing fragments
        reason = reason.rstrip(' ,;:')  # basic cleanup
        if len(reason) == 0:
            reason = "parse_truncated"
    else:
        # try looser: after "reason": take up to 120 chars
        loose = re.search(r'"reason"\s*:\s*"(.*)', candidate, flags=re.S)
        if loose:
            reason = loose.group(1)[:120].strip()
            reason = reason.rstrip(' ,;:')
        else:
            reason = "parse_error"

    # log a compact warning once (you already save raw_resp per row)
    print(f"[WARN] Failed to parse structured JSON strictly: {strict_err}. Falling back to regex. Raw head: {raw[:150]!r}")

    return label, reason, raw

# -------------------------
# 7. Processing loop (negative file only)
# -------------------------
def process_negative_sample():
    # load parquet
    print(f"Loading {INPUT_PARQUET} ...")
    df = pd.read_parquet(INPUT_PARQUET).sample(frac=1, random_state=42)
    print(f"Loaded {len(df)} rows.")
    # group by country_code for balanced sampling similar to original script
    groups = defaultdict(list)
    for _, row in df.iterrows():
        c = row.get("country_code") or "unknown"
        groups[c].append(row)

    results = []
    done = Counter()

    target_per_country = 250  # keep same default as your previous script

    while any(groups.values()):
        for country, rows in list(groups.items()):
            if not rows or done[country] >= target_per_country:
                continue
            row = rows.pop(0)
            prompt = build_prompt(row.get("title"), row.get("tags"))
            raw = call_gemini(prompt)
            label, reason, raw_full = parse_json_response(raw)

            results.append({
                "title": row.get("title", ""),
                "tags": row.get("tags", ""),
                "description": row.get("description", ""),
                "video_id": row.get("video_id", ""),
                "country_code": country,
                "label": label,
                "reason": reason,
                "raw_resp": raw_full
            })
            done[country] += 1

            time.sleep(RATE_LIMIT_SLEEP)

            if len(results) % SAVE_EVERY == 0:
                out = f"{OUT_PREFIX}.csv"
                pd.DataFrame(results).to_csv(out, index=False)
                print(f"[SAVE] {len(results)} rows → {out} (counts: {dict(done)})")

    out = f"{OUT_PREFIX}.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"✅ Finished: {len(results)} rows ({len(done)} countries) → {out}")

if __name__ == "__main__":
    print("Starting Gemini classification for negative sample (sampled_neg.parquet).")
    process_negative_sample()
