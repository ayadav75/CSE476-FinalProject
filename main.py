import os
import json
import re
import time
import requests
import threading 
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from ddgs import DDGS  

# Config
API_KEY = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "bens_model")

# Threading Config
MAX_WORKERS = 15
SEARCH_LOCK = threading.Lock() # Prevents DuckDuckGo bans
FILE_LOCK = threading.Lock()   # Prevents JSON corruption

# File Paths
INPUT_PATH = Path("cse_476_final_project_test_data.json") 
OUTPUT_PATH = Path("cse_476_final_project_answers.json")

# Modified API client

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

# Tool: Internet Search (DuckDuckGo) 
def internet_search(query: str, num_results: int = 3) -> str:
    """
    Performs a DuckDuckGo Search and returns a summary string.
    """
    # print(f"\n[Search] Query: {query}") 
    # To avoid getting banned I have used locks for searches
    with SEARCH_LOCK:
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

# Domain Classifier
# Asks the LLM to categorize the input into one of the 5 known domains

def classify_domain(user_input: str) -> str:
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
        # Strategy: Few-Shot Chain-of-Thought (CoT)
        
        system = (
            "You are a mathematical reasoning expert. Solve the problem step-by-step.\n\n"
            "RULES:\n"
            "1. Show your work/reasoning clearly before the final answer.\n"
            "2. If the problem involves calculations, double-check your arithmetic.\n"
            "3. If the answer is a fraction, simplify it. If it is a decimal, keep significant figures.\n"
            "4. CRITICAL: End your response strictly with '####' followed by the final answer.\n"
            "5. Do not include units in the final answer (after ####) unless explicitly asked.\n\n"
            "EXAMPLES:\n"
            "User: What is 15% of 80?\n"
            "Your answer: 15% is 0.15. 0.15 * 80 = 12. #### 12\n\n"
            "User: Solve for x: 2x + 5 = 15\n"
            "Your response: Subtract 5 from both sides: 2x = 10. Divide by 2: x = 5. #### 5\n\n"
            "User: Find the sum of the first 5 prime numbers.\n"
            "Your answer: Primes are 2, 3, 5, 7, 11. Sum = 2+3+5+7+11 = 28. #### 28"
        )
        
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_input}]
        
        raw_resp = call_llm(messages, temperature=0.1)
        
        # Delimiter extraction
        if "####" in raw_resp:
            return raw_resp.split("####")[-1].strip()
            
        # No delim
        lines = [line.strip() for line in raw_resp.split('\n') if line.strip()]
        if not lines:
            return "0" # Total failure fallback
            
        last_line = lines[-1]
        
        # Last line check
        if "answer" in last_line.lower():
            return last_line.split(":")[-1].strip()
            
        return last_line

    # Using Self-Correction method
    def solve_coding(self, user_input: str) -> str:
        # First pass
        system = (
            "You are a Python coding assistant. Complete the function provided.\n"
            "Rules: 1. Output ONLY code body. 2. No imports. 3. No function def signature. 4. Indent 4 spaces."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_input}]
        code_draft = call_llm(messages, temperature=0.0)
        
        # Second pass critique
        critique_prompt = (
            f"Here is a draft code snippet:\n{code_draft}\n\n"
            "Check this code against these rules:\n"
            "1. Does it include the 'def' line? (It should NOT).\n"
            "2. Does it include imports? (It should NOT).\n"
            "3. Is the indentation correct?\n"
            "Output the FIXED code only. If it was already correct, output it unchanged"
        )
        
        messages.append({"role": "assistant", "content": code_draft})
        messages.append({"role": "user", "content": critique_prompt})
        
        fixed_code = call_llm(messages, temperature=0.0)
        
        # Cleanup
        clean = fixed_code.replace("```python", "").replace("```", "").strip()
        lines = clean.split('\n')
        if lines and lines[0].strip().startswith("def "):
            clean = "\n".join(lines[1:])
        return clean

    def solve_planning(self, user_input: str) -> str:
        # Strategy: Chain of Thought (CoT) and some action mapping
        
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

        if len(user_input) > 500 or "context" in user_input.lower():
            return self._solve_comprehension(user_input)
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

        if "SEARCH" in decision:
            return self._solve_search_based(user_input)
        elif "MULTIPLE_CHOICE" in decision:
            return self._solve_multiple_choice(user_input)
        else:
            return self._solve_comprehension(user_input)

    def _solve_multiple_choice(self, user_input: str) -> str:
        # Strategy: Search then -> Reason then -> Select Exact String
        
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

    def _solve_search_based(self, user_input: str) -> str:
        # Direct Search 
        search_context = internet_search(user_input)
        
        # Reasoning & Extraction
        system = (
            "You are a specific fact extractor. "
            "Use the provided Search Results to answer the user's question.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Read the search results to find the specific entity (name, place, chemical, year, etc.).\n"
            "2. GRAMMAR CHECK: If the user asks 'it is also called what', ensure you identify the correct subject.\n"
            "3. Output format: First provide concise reasoning, then '####', then the FINAL SHORT ANSWER.\n"
            "   Example: 'Search results indicate the capital is Paris. #### Paris'\n"
            "4. If Search Results are empty or irrelevant, strictly use your own internal knowledge.\n"
            "5. If it is a yes or no answer, answer with a definite yes or no only."
        )
        
        augmented_input = (
            f"Question: {user_input}\n\n"
            f"Search Results:\n{search_context}\n\n"
            "Final Answer (Reasoning #### Entity):"
        )
        
        messages = [{"role": "system", "content": system}, {"role": "user", "content": augmented_input}]
        resp = call_llm(messages, temperature=0.0).strip()
        
        # Extraction of clean output
        if "####" in resp:
            answer = resp.split("####")[-1].strip()
        else:
            answer = resp.split('\n')[-1].strip()
            
        # Boolean Mapping
        lower_ans = answer.lower().replace(".", "")
        if lower_ans in ["yes", "correct", "true", "['Yes']", "['YES']"]: return "true"
        if lower_ans in ["no", "incorrect", "false", "['No']", "['NO']"]: return "false"
        
        return answer.strip(" .\"'")

    def _solve_comprehension(self, user_input: str) -> str:
        # Reasoning-Based Q&A
        safe_input = user_input
        if len(safe_input) > 30000:
            safe_input = safe_input[:30000] + "\n...[TRUNCATED]..."

        system = (
            "You are a Reading Comprehension expert. "
            "Your goal is to answer the question using ONLY the provided text.\n\n"
            "PROCESS:\n"
            "1. Analyze the question to identify the specific event, entity, or relationship requested.\n"
            "2. Scan the text for all relevant mentions (not just the first one).\n"
            "3. Reason about the context: If there are conflicting facts (e.g., a rejected treaty vs a ratified one), "
            "choose the one that best satisfies the specific phrasing of the question. Always answer logically and deterministically.\n"
            "4. Output the final short answer.\n\n"
            "FORMAT:\n"
            "Provide a brief reasoning line, then '####', then the FINAL ANSWER ENTITY.\n"
            "Example: 'The text explains that X rejected the treaty, but Y signed it. #### Y'"
        )
        
        messages = [{"role": "system", "content": system}, {"role": "user", "content": safe_input}]
        resp = call_llm(messages, temperature=0.0, max_tokens=512).strip()
        
        if "####" in resp:
            answer = resp.split("####")[-1].strip()
        else:
            answer = resp.split("\n")[-1].strip()
            if ":" in answer:
                answer = answer.split(":")[-1].strip()
            
        return answer.strip(" .\"'")
    
def validate_answer(answer_str: str) -> str:
    if not isinstance(answer_str, str):
        return str(answer_str)
    if len(answer_str) >= 5000:
        return answer_str[:4900] + "...[TRUNCATED]"
    return answer_str
    
# Worker function for threading arranges items
def process_item(index: int, item: Dict[str, Any]) -> tuple:
    solver = AgentStrategies()
    user_input = item.get("input", "")
    
    try:
        inferred_domain = classify_domain(user_input)
        
        if inferred_domain == "math": prediction = solver.solve_math(user_input)
        elif inferred_domain == "coding": prediction = solver.solve_coding(user_input)
        elif inferred_domain == "planning": prediction = solver.solve_planning(user_input)
        elif inferred_domain == "future_prediction": prediction = solver.solve_prediction(user_input)
        else: prediction = solver.solve_common_sense(user_input)
    except Exception as e:
        print(f"Error on item {index}: {e}")
        prediction = "Error"
        inferred_domain = "Error"

    prediction = validate_answer(prediction)
    print(f"ID: {index+1:<4} | Domain: {inferred_domain:<15} | Completed")
    
    return index, prediction


# MAIN EXECUTION
def main():
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found.")
        return

    # Load Data
    with open(INPUT_PATH, 'r', encoding="utf-8") as f:
        all_data = json.load(f)

    all_data = all_data[0:50]

    final_outputs = [None] * len(all_data)

    # Make json file recoverable 
    processed_indices = set()
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH, 'r', encoding="utf-8") as f:
                loaded = json.load(f)
                if len(loaded) <= len(all_data):
                    # Fill what we have so far
                    for i, obj in enumerate(loaded):
                        final_outputs[i] = obj
                        processed_indices.add(i)
                    print(f"Resuming... {len(processed_indices)} items already done.")
        except:
            print("Could not read existing file. Starting fresh.")

    # Identify which items need processing
    items_to_process = []
    for i, item in enumerate(all_data):
        if i not in processed_indices:
            items_to_process.append((i, item))

    if not items_to_process:
        print("All items completed.")
        return

    print(f"Starting threads with {MAX_WORKERS} workers...")
    start_time = time.perf_counter()
    completed_questions = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks
        future_to_idx = {executor.submit(process_item, idx, item): idx for idx, item in items_to_process}
        
        for future in as_completed(future_to_idx):
            idx, prediction = future.result()
            
            # Store result in the correct index position
            final_outputs[idx] = {"output": prediction}
            completed_questions += 1
            print("Progress: ", completed_questions)
            
            # Write File Safe
            with FILE_LOCK:
                save_data = [x for x in final_outputs if x is not None]
                
                with open(OUTPUT_PATH, 'w', encoding="utf-8") as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
    end_time = time.perf_counter()
    print("Execution Time: ", end_time-start_time)

    print(f"Done. Outputs saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()