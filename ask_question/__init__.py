
import json, logging
import azure.functions as func
import numpy as np
import pandas as pd
from utils import embed_texts, faiss_search, load_all_embeddings, BS, make_sas, azure_openai_answer, rate_limit_ok, SUPPORT_URL

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
    except Exception:
        body = {}
    question = body.get("text") or body.get("q") or ""
    conversation_id = body.get("conversationId") or body.get("threadId") or "default"

    if not question:
        return func.HttpResponse(json.dumps({"error":"missing 'text'"}), status_code=400, mimetype="application/json")

    ok, count = rate_limit_ok(conversation_id)
    if not ok:
        return func.HttpResponse(json.dumps({
            "answer": f"You’ve reached the 3-message limit for this session. Please open a support ticket: {SUPPORT_URL}",
            "citations": []
        }), mimetype="application/json")

    # Load index artifacts
    try:
        chunks, _ = load_all_embeddings()
    except Exception as e:
        return func.HttpResponse(json.dumps({"error":"index not built yet"}), status_code=503, mimetype="application/json")

    qvec = embed_texts([question])
    ids = faiss_search(qvec, topk=8)

    sel = chunks.iloc[ids]
    # compose context
    context = "\n---\n".join(sel["text"].tolist()[:4])
    answer = azure_openai_answer(question, context)

    # build citations with SAS
    cites = []
    for _, row in sel.head(3).iterrows():
        blob_path = row["doc"]
        sas = make_sas("sop-raw", blob_path.split("#_")[0])
        cites.append({"doc": blob_path, "url": sas})

    payload = {"answer": answer, "citations": cites}
    return func.HttpResponse(json.dumps(payload), mimetype="application/json")
