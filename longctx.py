"""What happens to inputs longer than the 4,096-token context of the Ollama d3 packages?

Builds a context of unrelated RAGTruth passages padded to a target length and puts one fact either
at the start or at the end. The response states that fact. If the model sees the fact, p("every claim
supported") should be high; if that part of the input is dropped, it should be low.
A control row has no fact at all (p should be low).
"""
import json, sys, urllib.request

URL = "http://localhost:11434/v1/systemone"
FACT = "The Halvorsen bridge in Tromsø was closed for repairs on 3 March 2019."
RESPONSE = "The Halvorsen bridge in Tromsø was closed for repairs on 3 March 2019."
Q = "Is every claim in the response supported by the context?"
filler = " ".join(r["context"] for r in json.load(open("data/sample.json"))
                  if isinstance(r["context"], str) and "Tromsø" not in r["context"])

def ask(model, context):
    body = {"model": model, "state": {"context": context, "response": RESPONSE},
            "questions": {"q": {"type": "noul", "instructions": Q}}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
        return d["answers"]["q"]["noul"], d.get("usage", {}).get("input_tokens"), None
    except urllib.error.HTTPError as e:
        return None, None, f"HTTP {e.code}: {e.read()[:200]!r}"
    except Exception as e:
        return None, None, repr(e)[:200]

for model in sys.argv[1:]:
    for words in (1500, 2500, 4000, 8000, 20000):  # ~1.3-1.5 tokens per word
        pad = " ".join(filler.split()[:words])
        for where, ctx in (("start", FACT + " " + pad), ("end", pad + " " + FACT), ("none", pad)):
            p, tok, err = ask(model, ctx)
            print(json.dumps({"model": model, "words": words, "fact": where, "p": p, "tokens": tok, "err": err}), flush=True)
