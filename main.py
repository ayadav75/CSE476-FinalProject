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

def call_llm(messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 2048) -> str:
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
        # pass query as positional argument
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
    
    # Validation
    valid_domains = ["math", "coding", "planning", "future_prediction", "common_sense"]
    for d in valid_domains:
        if d in response:
            return d
    return "common_sense" # Default 


# Solving Functions

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

    
    import re

    def solve_planning(self, user_input: str) -> str:
        # Strategy: Chain of Thought (CoT) + Action Mapping
        # We allow the model to "think" about the state to improve logic, then extract lines
        
        system = (
            "You are an expert PDDL planning engine.\n"
            "Your Goal: Generate the shortest valid plan to reach the goal state.\n\n"
            "INSTRUCTIONS:\n"
            "1. ANALYZE: Read the available actions and their conditions in the input.\n"
            "2. THINK: Briefly simulate the state changes step-by-step to find the solution.\n"
            "3. OUTPUT: After thinking, output the final plan as a list of actions.\n\n"
            "FORMATTING RULES:\n"
            "1. Use strict PDDL format: (action arg1 arg2)\n"
            "2. Use HYPHENS for standard actions: pick-up, put-down, unstack.\n"
            "3. Use the exact action names defined in the input's examples (e.g., if example uses 'lift', use 'lift').\n"
            "4. Do NOT use prepositions (from, to, at, on, into).\n"
            "5. Do NOT use object types (block, object, crate).\n"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input}
        ]
        
        # Higher amount of tokens for CoT
        raw_output = call_llm(messages, temperature=0.0, max_tokens=1024).strip()
        
        # Post processing
        clean_lines = []
        forbidden = {"from", "to", "at", "on", "top", "of", "into", "using", "the", "a", "an",
                     "object", "block", "crate", "pallet", "truck", "hoist", "depot", "distributor", 
                     "planet", "province"}
        
        for line in raw_output.splitlines():
            line = line.strip()
            
            # Filter: Valid plan lines must start with '(' and end with ')'
            if not (line.startswith("(") and line.endswith(")")):
                continue
                
            # Cleanup: Normalize and strip forbidden words
            line = line.lower().replace("pick up", "pick-up").replace("put down", "put-down")
            
            content = line.replace("(", "").replace(")", "")
            words = content.split()
            
            # Keep only logic identifiers
            filtered = [w for w in words if w not in forbidden]
            
            if filtered:
                clean_lines.append(f"({' '.join(filtered)})")
                
        return "\n".join(clean_lines)


    def solve_prediction(self, user_input: str) -> str:
        # Ask model to box the answer
        system = (
            "You are a forecaster. The user asks for a prediction.\n"
            "Your output must be wrapped in \\boxed{}.\n"
            "If multiple items, use a list format inside the box: \\boxed{['Item1', 'Item2']}.\n"
            "If a single item, just put the item: \\boxed{Yes} or \\boxed{10.5}."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_input}]
        raw_resp = call_llm(messages, temperature=0.1)

        content = raw_resp.strip()
        
        # Extract content from \boxed{...}
        match = re.search(r"\\boxed\{(.*?)\}", content)
        if match:
            content = match.group(1)
        
        content = content.strip()
        if content.startswith("[") and content.endswith("]"):
            return content
        
        # Clean to floats
        try:
            float(content)
            return f"[{content}]"
        except ValueError:
            pass

        # Check if it is a comma-separated list 
        if "," in content and len(content) < 20 and " " not in content.replace(", ", ""):
             items = [f"'{x.strip()}'" for x in content.split(",")]
             return f"[{', '.join(items)}]"

        # Correct formating
        clean_content = content.replace("'", "").replace('"', "")
        return f"['{clean_content}']"


    def solve_common_sense(self, user_input: str) -> str:
        # Decision
        decision_system = (
            "You are a routing agent. Analyze the user input.\n"
            "1. If the input contains a long text/passage and asks to answer 'using the context', output 'COMPREHENSION'.\n"
            "2. If the input provides a list of numbered (0, 1) or lettered (A, B) options and asks to select the best answer, output 'MULTIPLE_CHOICE'.\n"
            "3. If the input is a standalone question (e.g., 'Who is X?', 'Is Y true?'), output 'SEARCH'.\n"
            "Output ONLY: 'COMPREHENSION', 'MULTIPLE_CHOICE', or 'SEARCH'."
        )
        
        # Truncate for decision speed
        preview_input = user_input[:500] + ("..." if len(user_input) > 500 else "")
        
        msgs = [{"role": "system", "content": decision_system}, {"role": "user", "content": preview_input}]
        decision = call_llm(msgs, temperature=0.0, max_tokens=10).strip().upper()
        
        print(f"[Agent] Routing Decision: {decision}")

        if "COMPREHENSION" in decision:
            return self._solve_comprehension(user_input)
        elif "MULTIPLE_CHOICE" in decision:
            return self._solve_multiple_choice(user_input)
        else:
            return self._solve_search_based(user_input)

    def _solve_multiple_choice(self, user_input: str) -> str:
        # Strategy: Search -> Reason -> Select Exact String
        
        core_question = user_input.split('\n')[0]
        if len(core_question) < 10:
            core_question = user_input[:150]
            
        search_context = internet_search(core_question)
        
        system = (
            "You are a multiple-choice answering expert.\n"
            "You will be given a Question and a List of Options.\n"
            "Task: Select the BEST and most LOGICAL option that answers the question based on the search results.\n\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY the exact text as is of the selected Option.\n"
            "2. Do NOT include the option number (e.g., if option is '0) Apple', output 'Apple').\n"
            # "3. Do NOT write full sentences.\n"
            #"2. If search results are ambiguous, choose the most scientifically or historically accepted answer."
        )
        
        augmented_input = (
            f"Original Input:\n{user_input}\n\n"
            f"Search Evidence:\n{search_context}\n\n"
            "Final Answer (Option Text Only):"
        )
        
        messages = [{"role": "system", "content": system}, {"role": "user", "content": augmented_input}]
        resp = call_llm(messages, temperature=0.0).strip()
        # Cleanup
        cleaned_resp = re.sub(r'^[\d\w]+[\)\.]\s*', '', resp)
        
        return cleaned_resp.strip(" .\"'")


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

# Taken from evaluate_results.py and modified
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

def evaluate_single_item(item: Dict, prediction: str, inferred_domain: str) -> bool:
    expected = item.get("output")
    input_text = item.get("input", "")
    if expected is None: return False

    # Math
    if inferred_domain == "math":
        p_num = extract_number(prediction)
        e_num = extract_number(expected)
        if p_num is not None and e_num is not None:
            return abs(p_num - e_num) < 1e-6

    # Planning / Coding / Future
    if inferred_domain in ["planning", "coding", "future_prediction"]:
        p_norm = "".join(str(prediction).split()).lower().replace("['", "").replace("']", "")
        e_norm = "".join(str(expected).split()).lower().replace("['", "").replace("']", "")
        if e_norm in p_norm or p_norm == e_norm: return True

    # Common Sense - Boolean Handling
    p_str = str(prediction).lower().strip()
    e_str = str(expected).lower().strip()
    
    if p_str == e_str: 
        return True
        
    # If expected is specifically boolean-like
    if e_str in ["true", "false"]:
        # Map yes/no just in case logic slipped
        if p_str == "yes": p_str = "true"
        if p_str == "no": p_str = "false"
        return p_str == e_str

    # 4. Fallback: LLM Judge
    return llm_judge_check(input_text, prediction, expected)

# --- 5. Main Execution Loop ---

def main():
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found.")
        return

    # Load Data
    with open(INPUT_PATH, 'r', encoding="utf-8") as f:
        all_data = json.load(f)

    test_batch = all_data[995:] 

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

        # Evaluate 
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
        time.sleep(0.2)

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