import os
import requests
import json
import re
import time
from collections import Counter

# Config
API_KEY  = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")  
MODEL    = os.getenv("MODEL_NAME", "bens_model")

# Taken straight from Tutorial
def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 256,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            # try best-effort to surface error text
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}

def solve_with_cot(question: str):
    system_prompt = "You are a logical solver. Think step-by-step. End your response with 'FINAL ANSWER: <your answer>'."
    prompt = f"Question: {question}\n\nLet's think step by step."
    result = call_model_chat_completions(prompt, system=system_prompt, temperature=0.0)

    if result["ok"]:
        text = result["text"]
        if "FINAL ANSWER:" in text:
            return text.split("FINAL ANSWER")[-1].strip()
        return text
    return "ERROR"

def strategy_self_consistency(question: str, num_samples: int = 3) -> str:

    answers = []
    system_prompt = "Solve the problem. Reply ONLY with the final answer. No explanation."
    
    for _ in range(num_samples):
        r = call_model_chat_completions(question, system=system_prompt, temperature=0.7)
        text = r["text"] if r["ok"] else None
        if text:
            answers.append(text)
        time.sleep(0.1)

    if answers == []:
        return "ERROR"

    
    tally = Counter(answers)
    choice = tally.most_common(1)
    winner = choice[0][0]
    return winner

def strategy_refinement(question: str) -> str:
    draft_result = call_model_chat_completions(question, temperature=0.0)
    if not draft_result["ok"]: return "ERROR"
    draft = draft_result["text"]
    
    critique_prompt = (
        f"Question: {question}\n"
        f"Proposed Answer: {draft}\n"
        "Critique this answer for logical or arithmetic errors. "
        "Then provide the corrected FINAL ANSWER only."
    )
    final_res = call_model_chat_completions(critique_prompt, system="You are a strict reviewer.", temperature=0.0)
    
    if final_res["ok"]:
        return final_res["text"]
    return draft

