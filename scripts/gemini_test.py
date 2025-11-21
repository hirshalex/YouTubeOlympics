import os
import json
import re
import google.generativeai as genai
from typing import Optional

# Configure your Gemini API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Load your prompt template
with open("scripts/groq_prompt_examples.md", "r", encoding="utf-8") as f:
    base_prompt = f.read().strip()

sample = {
    "title": "Makeup Tutorial for Gamers",
    "tags": "makeup, gaming, tutorial"
}

prompt = base_prompt.replace("<<title>>", sample['title']).replace("<<tags>>", sample['tags'])

# choose model you have access to
model = genai.GenerativeModel("gemini-2.5-flash")  # or "gemini-2.5-pro"

# tweak generation_config to be explicit
generation_config = genai.types.GenerationConfig(
    temperature=0.0,
    max_output_tokens=256,
    response_mime_type="text/plain",  # ask for plain text
)

response = model.generate_content(
    prompt,
    generation_config=generation_config
)

# Debug: print response repr so you can inspect the raw object
print("FULL RESPONSE (repr):\n", repr(response), "\n\n")

# Attempt robust extraction of text from multiple possible response shapes
def extract_text_from_response(resp) -> Optional[str]:
    # 1) try quick accessor (may raise ValueError if no Part exists)
    try:
        txt = resp.text
        if txt:
            return txt
    except Exception as e:
        # quick accessor failed; we'll attempt manual extraction
        print("quick .text accessor failed:", e)

    # 2) look into resp.candidates[] -> candidate.content (list of parts)
    cand_list = getattr(resp, "candidates", None)
    if cand_list:
        for i, cand in enumerate(cand_list):
            # print candidate-level info for debugging
            finish_reason = getattr(cand, "finish_reason", None)
            print(f"candidate[{i}] finish_reason:", finish_reason)
            content = getattr(cand, "content", None)
            if not content:
                # sometimes candidate has nested structure; try to stringify
                try:
                    print(f"candidate[{i}] (no .content): {cand}")
                except Exception:
                    pass
                continue

            # content can be list of dicts or Part objects
            for j, part in enumerate(content):
                # if part is dict-like
                if isinstance(part, dict):
                    # common key is 'text'
                    if "text" in part and part["text"]:
                        return part["text"]
                    # some shapes may keep text under 'value' or similar
                    if "value" in part and isinstance(part["value"], str):
                        return part["value"]

                # if part is an object with .text attribute
                part_text = getattr(part, "text", None)
                if part_text:
                    return part_text

                # try stringifying the part if nothing else
                try:
                    s = str(part)
                    if s and len(s) > 0:
                        # be conservative: only return if it's plausible text (not object repr)
                        if len(s) > 10:
                            return s
                except Exception:
                    pass

    # 3) some responses include resp.output or resp.result
    for candidate_attr in ("output", "result", "resp"):
        value = getattr(resp, candidate_attr, None)
        if value:
            try:
                s = str(value)
                if s and len(s) > 0:
                    return s
            except Exception:
                pass

    # 4) ultimate fallback: str(response)
    try:
        return str(resp)
    except Exception:
        return None

raw = extract_text_from_response(response)
if raw is None:
    print("Failed to extract text from response. Full object printed above for debugging.")
else:
    raw = raw.strip()
    print("RAW RESPONSE:\n", raw)

    # Now extract JSON block (same as your GROQ code)
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            print("\nPARSED JSON:", json.dumps(data, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print("\nFound JSON-like block but failed to parse:", e)
            print("JSON block:\n", m.group(0))
    else:
        print("\nNo JSON found in the model output.")