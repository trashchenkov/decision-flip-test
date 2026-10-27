"""Ask a decision model the same question in opposite wordings over the RAGTruth sample.

Appends one JSON line per (model, wording, row) to results/<out>.jsonl; skips rows already done.
Usage:
  python run.py ollama:d3:4b [ollama:d3:9b ...] [--limit 500] [--out mine]
  python run.py laya strands-2b              # needs `pip install laya strands-decider`
To test your own model, add a branch to load() that returns ask(state, question_text) -> p_yes.
"""
import json, os, sys, time, urllib.request

WORDINGS = {
    "A_en": "Is every claim in the response supported by the context?",
    "B_en": "Does the response contain any claim that is not supported by the context?",
    "IRR_fr": "Is the response written in French?",
}
args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
limit = opt("--limit", None); limit = int(limit) if limit else None
OUT = f"results/{opt('--out', 'mine')}.jsonl"
rows = json.load(open("data/sample.json"))[:limit]
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        d = json.loads(l); done.add((d["model"], d["wording"], d["id"]))

def load(name):
    if name.startswith("ollama:"):
        model = name[len("ollama:"):]
        url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/v1/systemone"
        def ask(state, text):
            body = {"model": model, "state": state, "questions": {"q": {"type": "noul", "instructions": text}}}
            req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=300))["answers"]["q"]["noul"]
        return ask
    if name == "laya":
        import laya
        ag = laya.load("convaiinnovations/laya", subfolder="multilingual")
        return lambda s, t: ag.predict(s, {"q": {"type": "noul", "instructions": t}}, max_len=8192)["answers"]["q"]["noul"]
    if name == "strands-2b":
        from strands_decider.infer import load_engine
        eng = load_engine("StrandsAgents/strands-decider-2B-hobson-v19")
        return lambda s, t: eng.ask(s, {"q": {"type": "noul", "instructions": t}}).answers["q"].noul
    raise ValueError(f"unknown model {name}")

os.makedirs("results", exist_ok=True)
for name in args:
    ask = load(name)
    with open(OUT, "a") as f:
        for wkey, text in WORDINGS.items():
            t0 = time.time(); n = 0
            for r in rows:
                if (name, wkey, r["id"]) in done:
                    continue
                try:
                    p, err = ask({"context": r["context"], "response": r["response"]}, text), None
                except Exception as e:
                    p, err = None, repr(e)[:300]
                f.write(json.dumps({"model": name, "wording": wkey, "id": r["id"], "p": p, "err": err}) + "\n")
                f.flush(); n += 1
            print(f"{name} {wkey}: {n} rows in {time.time()-t0:.0f}s", flush=True)
