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

# --- Helpers ---
def extract_answer(text: str) -> str:
    """
    Finds the answer after 'FINAL ANSWER:'.
    If not found, tries to grab the last number.
    """
    if not text: return ""
    
    # 1. Try explicit delimiter
    if "FINAL ANSWER:" in text:
        return text.split("FINAL ANSWER:")[-1].strip()
    
    # 2. Fallback: If the text is short (< 20 chars), it's probably just the answer
    if len(text) < 20:
        return text

    # 3. Fallback: Regex to find the last number in the text
    # (This helps if the model forgets the delimiter but does the math)
    matches = re.findall(r'-?\d+\.?\d*', text)
    if matches:
        return matches[-1]
        
    return text

def clean_text(text: str) -> str:
    """
    Normalizes text for grading:
    1. Extracts the relevant part.
    2. Lowercases and removes non-alphanumeric chars.
    """
    # First, get the specific answer part
    text = extract_answer(text)
    
    # Then normalize (remove punctuation, spaces, etc)
    text = re.sub(r"[^a-zA-Z0-9.\-]", "", text)
    return text.strip().lower()

# --- Strategies ---

def solve_with_cot(question: str) -> str:
    """Strategy 1: Chain of Thought"""
    system_prompt = "You are a logical solver. Think step-by-step. End your response with 'FINAL ANSWER: <your answer>'."
    prompt = f"Question: {question}\n\nLet's think step by step."
    
    response = call_model_chat_completions(prompt, system=system_prompt, temperature=0.0)

    if response["ok"]:
        return clean_text(response["text"])
    return "ERROR"

def solve_with_consistency(question: str, num_samples: int = 3) -> str:
    """Strategy 2: Self-Consistency"""
    answers = []
    # CHANGED: Explicitly ask for FINAL ANSWER format to fix the verbose output issue
    system_prompt = "You are a math solver. Solve step-by-step. YOU MUST END YOUR RESPONSE WITH 'FINAL ANSWER: <answer>'."
    
    for _ in range(num_samples):
        response = call_model_chat_completions(question, system=system_prompt, temperature=0.7)
        if response["ok"]:
            # Extract and clean immediately
            ans = clean_text(response["text"])
            if ans: answers.append(ans)
        time.sleep(0.1)

    if not answers: return "ERROR"
    
    # Voting
    counts = Counter(answers)
    return counts.most_common(1)[0][0]

def solve_with_refinement(question: str) -> str:
    """Strategy 3: Refinement"""
    # Draft
    draft_resp = call_model_chat_completions(question, temperature=0.0)
    if not draft_resp["ok"]: return "ERROR"
    draft = draft_resp["text"]
    
    # Critique
    critique_prompt = (
        f"Question: {question}\n"
        f"Proposed Answer: {draft}\n"
        "Check for errors. Then provide the corrected answer.\n"
        "YOU MUST END YOUR RESPONSE WITH 'FINAL ANSWER: <answer>'."
    )
    final_resp = call_model_chat_completions(critique_prompt, system="You are a strict reviewer.", temperature=0.0)
    
    if final_resp["ok"]:
        return clean_text(final_resp["text"])
    return clean_text(draft)

# --- Agent Router ---
def run_agent(question: str) -> dict:
    """Decides which strategy to use based on the question type."""
    start_time = time.time()
    
    # 1. Classify the problem
    router_prompt = (
        f"Question: {question}\n\n"
        "Classify this problem:\n"
        "1. MATH (Calculation/Algebra) -> Use Self-Consistency\n"
        "2. LOGIC (Reasoning) -> Use Refinement\n"
        "3. SIMPLE (Direct Fact) -> Use CoT\n"
        "Reply with exactly: MATH, LOGIC, or SIMPLE"
    )
    
    router_resp = call_model_chat_completions(router_prompt, system="You are a classifier.", temperature=0.0)
    category = router_resp["text"].strip().upper() if router_resp["ok"] else "SIMPLE"
    
    # 2. Dispatch to strategy
    if "MATH" in category:
        answer = solve_with_consistency(question)
        final_cat = "MATH (Consistency)"
    elif "LOGIC" in category:
        answer = solve_with_refinement(question)
        final_cat = "LOGIC (Refinement)"
    else:
        answer = solve_with_cot(question)
        final_cat = "SIMPLE (CoT)"
        
    duration = round(time.time() - start_time, 2)
    
    return {
        "answer": answer,
        "category": final_cat,
        "time": duration
    }

# --- Grading Helper ---
def normalize_for_grading(text):
    """Normalizes text for comparison (lowercase, alphanumeric only)."""
    if not text: return ""
    text = str(text).lower()
    return re.sub(r"[^a-z0-9]", "", text)

# --- Main Execution ---
if __name__ == "__main__":
    INPUT_FILE = "cse476_final_project_dev_data.json"
    OUTPUT_FILE = "outputs.json"
    TEST_LIMIT = 5 # Set to None to run all

    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} missing.")
        exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    
    dataset = dev_data[:TEST_LIMIT] if TEST_LIMIT else dev_data
    print(f"Loaded {len(dev_data)} questions. Running {len(dataset)}.")

    # 2. Run Loop
    results = []
    correct_count = 0
    
    for i, item in enumerate(dataset):
        question = item.get("input")
        expected = item.get("output")
        domain = item.get("domain", "unknown")

        print(f"\n--- Question {i+1} [{domain}] ---")
        print(f"Input: {question[:100]}...")

        try:
            # Call the router, not the raw API
            output = run_agent(question)
            prediction = output["answer"]
            strategy = output["category"]
            elapsed = output["time"]
        except Exception as e:
            print(f"Agent Error: {e}")
            prediction, strategy, elapsed = "ERROR", "FAILED", 0.0

        # 3. Grade
        norm_expected = normalize_for_grading(expected)
        norm_pred = normalize_for_grading(prediction)
        is_correct = (norm_expected == norm_pred)
        
        if is_correct:
            correct_count += 1
            icon = "✅ PASS"
        else:
            icon = "❌ FAIL"

        print(f"   Strategy: {strategy}")
        print(f"   Time:     {elapsed}s")
        print(f"   Expected: {expected}")
        print(f"   Got:      {prediction}")
        print(f"   Result:   {icon}")

        results.append({
            "id": i,
            "input": question,
            "expected": expected,
            "prediction": prediction,
            "correct": is_correct,
            "strategy": strategy,
            "domain": domain
        })

    # 4. Report & Save
    total = len(dataset)
    accuracy = (correct_count / total * 100) if total > 0 else 0
    
    print("\n" + "="*30)
    print(f"FINAL ACCURACY: {accuracy:.2f}% ({correct_count}/{total})")
    print("="*30)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {OUTPUT_FILE}")