# CSE476-FinalProject Anshuman Yadav

This agent implements three core inference-time algorithms:

1.  **Dynamic Routing (Classifier)**
    *   Level 1: The agent first analyzes the input to classify it into one of 5 domains: Math, Planning, Future Predictions or Common Sense
    *   Level 2: Within Common Sense the questions get classified into Comprehension, Multiple Choice, or Search. 

2.  **RAG/ Tool Use**
    *   **Domain**: Common Sense based questions
    *   **Mechanism for Search**: The agent checks what type of common sense question is presented during the routing. If an only if, the agent recognizes questions requiring external knowledge, the search function is called. I have used **DuckDuckGo**'s free API to search the internet. Then the model uses information from the internet as context to get the best answer.

3.  **Chain-of-Thought with Logic Extraction**
    *   **Domain**: Math & Planning
    *   **Mechanism**:
        *   **Math**: The model is instructed with few shot CoT to solve step-by-step but it must end with a specific delimiter (`#### <Answer>`). The agent then extracts the final value.
        *   **Planning**: The model simulates thinking before generating the plan. A Python-based regex filter then extracts only the valid actions, discarding the reasoning text to ensure correct formatting.
4. **Self Correction**
   * **Domain:** Coding
   * The model generates a first draft of the code. Within the same function this first draft is fed in another call to check if the format requested and the overall correctness of the prompt is maintained or not. 

---

## Installation

### Dependencies:
Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

---

## Usage
1.  **Place Data File**: Ensure the test data file is in the same directory as the script.
    *   Input Filename: `cse_476_final_project_test_data.json` or use `cse476_final_project_dev_data.json`

2.  **Run the Agent**:
    ```bash
    python main.py
    ```

3.  **View Results**:
    *   The script will print progress to the console and Domain detections.
    *   The final outputs will be saved to: `cse_476_final_project_answers.json`.
