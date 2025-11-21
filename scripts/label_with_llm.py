#!/usr/bin/env python3
# scripts/label_with_llm_stream.py
"""
Streaming LLM labeling for a parquet input. Writes incremental CSV output.
Features:
 - streaming read so it won't OOM
 - incremental flush every `save_every` rows
 - resilient model loading (GPU 4-bit if available, otherwise CPU)
 - exponential backoff on generation errors
 - simple parsing and label normalization
 - supports --start and --maxrows for sharded runs
"""
import argparse, os, time, csv, json, sys
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import pyarrow.dataset as ds
import pyarrow as pa
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="parquet input (can be large)")
parser.add_argument("--output", required=True, help="csv output (appended incrementally)")
parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HF model id")
parser.add_argument("--prompt_file", default="scripts/labeling_prompt_examples.md")
parser.add_argument("--maxrows", type=int, default=None)
parser.add_argument("--start", type=int, default=0, help="skip this many rows (sharding)")
parser.add_argument("--batch_size", type=int, default=1, help="rows per LLM call (1 recommended for clarity)")
parser.add_argument("--save_every", type=int, default=100, help="flush CSV every N labeled rows")
parser.add_argument("--throttle", type=float, default=0.02, help="seconds sleep between calls to be polite")
parser.add_argument("--device", type=int, default=0, help="preferred GPU device index (if GPU available)")
args = parser.parse_args()

# load few-shot prompt
few_shot = ""
if os.path.exists(args.prompt_file):
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        few_shot = f.read()
else:
    few_shot = "Title: Simone Biles wins Olympic vault final\nTags: olympics, gymnastics\nLabel: olympic\nReason: about the Olympics.\n\n"

# helper: machine-friendly instruction appended to prompt
FORMAT_INSTRUCTIONS = (
    "\nAnswer using EXACTLY the following format (no extra text):\n"
    "Label: <one of olympic | other_sport | non_sport>\n"
    "Reason: <one short sentence>\n"
)

prompt_prefix = few_shot + "\n"

# robust model loading
print(f"[{datetime.utcnow().isoformat()}] Loading model {args.model}", file=sys.stderr, flush=True)
tokenizer = None
model = None
pipe = None
try:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # try GPU + quant (bitsandbytes) first
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, device_map="cpu",  torch_dtype=torch.float16,  low_cpu_mem_usage=True
        )
        # pipeline: device set to 0 if CUDA visible; else pipeline will use the model's device_map
        pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)
        print("[INFO] Loaded quantized model with device_map='auto'.", file=sys.stderr, flush=True)
    except Exception as e:
        print("[WARN] GPU/4bit load failed, falling back to CPU or non-quantized. Error:", e, file=sys.stderr, flush=True)
        # fallback: load on CPU (no quant) - slower but safer
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, device_map="cpu")
        pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)
        print("[INFO] Loaded model on CPU.", file=sys.stderr, flush=True)
except Exception as e:
    print("[ERROR] Failed to load model/tokenizer:", e, file=sys.stderr)
    raise

# helper to build prompt for a row dict
def build_prompt(row):
    title = str(row.get("title","") or "")[:800]
    tags = str(row.get("tags","") or "")[:400]
    desc = str(row.get("description","") or "")[:800]
    text = f"{prompt_prefix}Title: {title}\nTags: {tags}\nDescription: {desc}\n\n{FORMAT_INSTRUCTIONS}"
    return text

# parsing the response into label + reason
VALID_LABELS = {"olympic","other_sport","non_sport"}
def parse_response(resp_text):
    label = "non_sport"
    reason = ""
    for line in resp_text.splitlines():
        line = line.strip()
        if line.lower().startswith("label:"):
            parts = line.split(":",1)
            if len(parts) > 1:
                candidate = parts[1].strip().split()[0].lower()
                if candidate in VALID_LABELS:
                    label = candidate
            break
    # Reason line (optional)
    for line in resp_text.splitlines():
        if line.lower().strip().startswith("reason:"):
            reason = line.split(":",1)[1].strip()
            break
    return label, reason, resp_text

# streaming read from parquet dataset
dataset = ds.dataset(args.input, format="parquet")
# choose columns we need if present; keep identifiers to inspect later
cols = [c for c in ["title","tags","description","channelTitle","video_id","country_code","publish_time"] if c in dataset.schema.names]
scanner = dataset.scanner(columns=cols or None, batch_size=1)  # we use batch_size=1 for predictable mapping to input rows

# prepare output CSV (append mode)
os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
header = ["index","title","tags","description","video_id","country_code","label","reason","raw_resp"]
if not os.path.exists(args.output):
    f_out = open(args.output, "w", encoding="utf8", newline="")
    writer = csv.DictWriter(f_out, fieldnames=header)
    writer.writeheader()
    f_out.flush()
else:
    f_out = open(args.output, "a", encoding="utf8", newline="")
    writer = csv.DictWriter(f_out, fieldnames=header)

# iterate and label
labeled = 0
skipped = 0
row_idx = 0
start = args.start
maxrows = args.maxrows
it = scanner.to_batches()
# skip rows up to start
while row_idx < start:
    try:
        next(it)
        row_idx += 1
    except StopIteration:
        break

def safe_generate(prompt_text, max_retries=4):
    delay = 1.0
    for attempt in range(max_retries):
        try:
            out = pipe(prompt_text, max_new_tokens=120, do_sample=False)
            # pipeline returns a list of dicts
            if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
                return out[0]["generated_text"]
            # fallback if structure different
            return str(out)
        except Exception as e:
            print(f"[WARN] generation error (attempt {attempt+1}): {e}", file=sys.stderr, flush=True)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("LLM generation failed after retries")

try:
    while True:
        if maxrows is not None and labeled >= maxrows:
            break
        try:
            batch = next(it)
        except StopIteration:
            break
        n = batch.num_rows
        # convert only first row (we set batch_size=1)
        try:
            # extract row fields without full pandas conversion
            row = {}
            for i, name in enumerate(batch.schema.names):
                try:
                    row[name] = batch.column(i).slice(0,1).to_pylist()[0]
                except Exception:
                    row[name] = None
        except Exception as e:
            skipped += 1
            row_idx += 1
            continue

        if row_idx < start:
            row_idx += 1
            continue

        prompt = build_prompt(row)
        try:
            resp = safe_generate(prompt)
        except Exception as e:
            print("[ERROR] generation failed permanently:", e, file=sys.stderr)
            # write a fallback row marking failure
            writer.writerow({
                "index": row_idx,
                "title": row.get("title",""),
                "tags": row.get("tags",""),
                "description": row.get("description",""),
                "video_id": row.get("video_id",""),
                "country_code": row.get("country_code",""),
                "label": "non_sport",
                "reason": "generation_failed",
                "raw_resp": ""
            })
            f_out.flush()
            labeled += 1
            row_idx += 1
            continue

        label, reason, raw = parse_response(resp)
        writer.writerow({
            "index": row_idx,
            "title": row.get("title",""),
            "tags": row.get("tags",""),
            "description": row.get("description",""),
            "video_id": row.get("video_id",""),
            "country_code": row.get("country_code",""),
            "label": label,
            "reason": reason,
            "raw_resp": raw
        })
        labeled += 1
        row_idx += 1

        if labeled % args.save_every == 0:
            f_out.flush()
            print(f"[{datetime.utcnow().isoformat()}] Labeled {labeled} rows (index {row_idx})", file=sys.stderr, flush=True)

        time.sleep(args.throttle)

finally:
    f_out.flush()
    f_out.close()
    print(f"[{datetime.utcnow().isoformat()}] Done. Labeled={labeled}, skipped={skipped}", file=sys.stderr)
