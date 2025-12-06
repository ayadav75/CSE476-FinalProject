# CSE476-FinalProject

This agent implements three core inference-time algorithms:

1.  **Dynamic Routing (Classifier)**
    *   The agent first analyzes the input to classify it into one of 5 domains.
    *   This allows the application of specific constraints that would otherwise confuse a generalist model.

2.  **Retrieval-Augmented Generation (RAG) / Tool Use**
    *   **Domain**: For Common Sense based questions
    *   **Mechanism**: The agent checks what type of common sense question is presented. If an only if, the agent recognizes questions requiring external knowledge: it generates a search query. I have used **DuckDuckGo**'s free API to search the internet. Then the model uses information from the internet to get the best answer.

3.  **Chain-of-Thought with Logic Extraction**
    *   **Domain**: Math & Planning.
    *   **Mechanism**:
        *   **Math**: The model is instructed to solve step-by-step but it must end with a specific delimiter (`#### <Answer>`). The agent then extracts the final value.
        *   **Planning**: The model simulates thinking before generating the plan. A Python-based regex filter then extracts only the valid actions, discarding the reasoning text to ensure correct formatting.

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
    *   Input Filename: `cse_476_final_project_test_data.json` or `cse476_final_project_dev_data.json`

2.  **Run the Agent**:
    ```bash
    python main.py
    ```

3.  **View Results**:
    *   The script will print progress to the console and Domain detections.
    *   The final outputs will be saved to: `cse_476_final_project_answers.json`.
