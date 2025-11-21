import groq, json, re, os

client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])

# Load your prompt template
with open("scripts/groq_prompt_examples.md", "r", encoding="utf-8") as f:
    base_prompt = f.read().strip()

# Example video - NO DESCRIPTION
sample = {
    "title": "Makeup Tutorial for Gamers",
    "tags": "makeup, gaming, tutorial"
}
prompt = base_prompt.replace("<<title>>", sample['title']).replace("<<tags>>", sample['tags'])

resp = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
    max_tokens=60,
)
raw = resp.choices[0].message.content.strip()
print("RAW RESPONSE:\n", raw)

m = re.search(r"\{.*\}", raw, flags=re.S)
if m:
    data = json.loads(m.group(0))
    print("\nPARSED:", data)
else:
    print("\nNo JSON found.")