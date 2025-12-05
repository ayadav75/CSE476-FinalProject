import os
import json
import re
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

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

def classify_domain(user_input: str) -> str:
    """
    Asks the LLM to categorize the input into one of the 5 known domains.
    """
    system_prompt = (
        "You are a classification agent. "
        "Analyze the user input and categorize it into exactly one of these domains:\n"
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

