import os
import json
import re
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from ddgs import DDGS  

# --- Configuration ---
API_KEY = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "bens_model")

# File Paths
INPUT_PATH = Path("cse476_final_project_dev_data.json") 
OUTPUT_PATH = Path("cse_476_final_project_answers.json")

# --- 1. API Client ---

def call_llm(messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """Standard API wrapper with retry logic."""
    url = f"{API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens
    }
    
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                return content
            else:
                # Print error but don't crash, try again
                pass 
        except Exception:
            pass
        time.sleep(1)
    return ""
# --- Tool: Internet Search (DuckDuckGo) ---
def internet_search(query: str, num_results: int = 3) -> str:
    """
    Performs a DuckDuckGo Search and returns a summary string.
    """
    # print(f"\n[Search] Query: {query}") 
    try:
        results = list(DDGS().text(query, max_results=num_results))
        
        if not results:
            print("[Search] No results found.")
            return ""
        
        context_pieces = []
        for item in results:
            title = item.get("title", "No Title")
            snippet = item.get("body", "No Snippet")
            context_pieces.append(f"Source: {title}\nContent: {snippet}\n")
        
        context_str = "\n---\n".join(context_pieces)
        # print(f"[Search] Context Length: {len(context_str)} chars") 
        return context_str
        
    except Exception as e:
        print(f"[Search] Error: {e}")
        return ""

# --- 2. The Domain Classifier ---

def classify_domain(user_input: str) -> str:
    """
    Asks the LLM to categorize the input into one of the 5 known domains.
    """
    system_prompt = (
        "You are a classification agent. "
        "Analyze the user input and categorize it into exactly one of these domains:\n"
        "If the input involves words that indicate that the user is in first person planning to do an activity, like I am or I plan... then it is a planning domain.\n"
        "1. math\n"
        "2. coding\n"
        "3. planning\n"
        "4. future_prediction\n"
        "5. common_sense\n\n"
        "Return ONLY the domain name (lowercase). Do not explain."
    )
    
    # Truncate input for classification to save tokens/time if it's huge
    short_input = user_input[:1000] 
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": short_input}
    ]
    
    response = call_llm(messages, temperature=0.0, max_tokens=10).strip().lower()
    
    # Validation/Fallback
    valid_domains = ["math", "coding", "planning", "future_prediction", "common_sense"]
    for d in valid_domains:
        if d in response:
            return d
    return "common_sense" # Default fallback

class AgentStrategies:
    
    def solve_math(self, user_input: str) -> str:
        # CoT + Strict Number Extraction Format
        system = (
            "You are a math expert. Solve the problem step-by-step.\n"
            "At the very end, output the final numeric answer strictly in this format:\n"
            "#### <NUMBER>\n"
            "Example: #### 42"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_input}]
        return call_llm(messages, temperature=0.2)

    def solve_coding(self, user_input: str) -> str:
        # Code Completion Mode
        system = (
            "You are a Python coding assistant. "
            "Complete the function or code block provided.\n"
            "1. Output ONLY the code body.\n"
            "2. Do NOT output 'def function_name...'.\n"
            "3. Do NOT wrap in markdown (no ```).\n"
            "4. Follow strict indentation (4 spaces)."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_input}]
        resp = call_llm(messages, temperature=0.0)
        
        # Cleanup: Remove markdown and def lines
        clean = resp.replace("```python", "").replace("```", "").strip()
        lines = clean.split('\n')
        if lines and lines[0].strip().startswith("def "):
            clean = "\n".join(lines[1:])
        return clean

def normalize_text(text: str) -> str:
    """Basic text normalization."""
    if not text: return ""
    return str(text).strip().lower()

def extract_number(text: str) -> Optional[float]:
    """Extracts numbers for math comparison."""
    if not text: return None
    matches = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)", str(text))
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None

def llm_judge_check(question: str, prediction: str, expected: str) -> bool:
    """Uses the LLM to judge semantic correctness."""
    prompt = (
        "You are a strict grader. Compare the PREDICTION to the EXPECTED ANSWER.\n"
        "Return the word 'TRUE' if the prediction means the same thing, contains the correct code logic, or is a valid equivalent.\n"
        "Return 'FALSE' otherwise.\n\n"
        f"QUESTION START: {question[:200]}...\n"
        f"EXPECTED: {expected}\n"
        f"PREDICTION: {prediction}\n\n"
        "Verdict (TRUE/FALSE):"
    )
    # We use a quick call for judging
    verdict = call_llm([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=5)
    return "TRUE" in verdict.upper()
    
def main():
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found.")
        return

    # Load Data
    with open(INPUT_PATH, 'r', encoding="utf-8") as f:
        all_data = json.load(f)

    test_batch = all_data[0:5] 

    solver = AgentStrategies()
    results = []
    
    correct_count = 0
    total_count = 0

    print(f"{'ID':<4} | {'TRUE DOM':<10} | {'INF DOM':<10} | {'JUDGE':<8} | {'DETAILS'}")
    print("-" * 100)

    for i, item in enumerate(test_batch):
        user_input = item.get("input", "")
        true_domain = item.get("domain", "N/A")

        # 1. Classify Domain
        inferred_domain = classify_domain(user_input)

        # 2. Route to Solver
        if inferred_domain == "math":
            prediction = solver.solve_math(user_input)
        elif inferred_domain == "coding":
            prediction = solver.solve_coding(user_input)
        elif inferred_domain == "planning":
            prediction = solver.solve_planning(user_input)
        elif inferred_domain == "future_prediction":
            prediction = solver.solve_prediction(user_input)
        else:
            prediction = solver.solve_common_sense(user_input)

        # Store result
        results.append({
            "input": user_input,
            "output": prediction
        })

        # 3. Evaluate (LLM as Judge + Heuristics)
        # Only evaluate if we have the 'output' key (Dev Data)
        if "output" in item:
            total_count += 1
            is_pass = evaluate_single_item(item, prediction, inferred_domain)
            
            judge_str = "✅ PASS" if is_pass else "❌ FAIL"
            if is_pass: correct_count += 1
            
            # Truncate for display
            clean_pred = str(prediction).replace("\n", " ")[:30]
            print(f"{i:<4} | {true_domain:<10} | {inferred_domain:<10} | {judge_str:<8} | Out: {clean_pred}...")
        else:
            print(f"{i:<4} | {true_domain:<10} | {inferred_domain:<10} | {'SKIP':<8} | (No ground truth)")

        # Rate limit
        time.sleep(0.5)

    # Final Stats
    if total_count > 0:
        print("-" * 100)
        print(f"Accuracy: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")

    # Save to JSON
    with open(OUTPUT_PATH, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nFull outputs saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()