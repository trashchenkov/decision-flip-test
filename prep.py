"""Download RAGTruth and build a fixed random sample of test rows: context, response, hallucinated label."""
import json, os, random, urllib.request

BASE = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/"
SEED, N = 42, 1000
os.makedirs("data", exist_ok=True)
for name in ("response.jsonl", "source_info.jsonl"):
    if not os.path.exists(f"data/{name}"):
        urllib.request.urlretrieve(BASE + name, f"data/{name}")

src = {}
for line in open("data/source_info.jsonl"):
    s = json.loads(line)
    src[s["source_id"]] = s

def context_text(s):
    info = s["source_info"]
    if s["task_type"] == "QA":
        return f"Question: {info['question']}\nPassages:\n{info['passages']}"
    if s["task_type"] == "Data2txt":
        return json.dumps(info, ensure_ascii=False)
    return info

rows = [json.loads(l) for l in open("data/response.jsonl")]
test = [r for r in rows if r["split"] == "test"]
random.Random(SEED).shuffle(test)
out = []
for r in test[:N]:
    s = src[r["source_id"]]
    out.append({"id": r["id"], "task": s["task_type"], "gen_model": r["model"],
                "context": context_text(s), "response": r["response"],
                "hallucinated": int(bool(r["labels"]))})
json.dump(out, open("data/sample.json", "w"), ensure_ascii=False)
print(len(out), sum(o["hallucinated"] for o in out), "hallucinated")
