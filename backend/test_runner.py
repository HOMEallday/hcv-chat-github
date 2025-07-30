import requests
import json
import time
import os
from dotenv import load_dotenv


load_dotenv()


# --- Configuration ---
API_URL = "http://127.0.0.1:8000/test-performance"
API_KEY = os.getenv("PERFORMANCE_TEST_API_KEY")
if not API_KEY:
    raise ValueError("PERFORMANCE_TEST_API_KEY not found in .env file. Please set it.")

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

# --- Grid Search Parameters ---

# 1. Define the models you want to test
MODELS_TO_TEST = [
    "gemini-2.5-flash"
]

# 2. MODIFIED: Define tests that pair a specific prompt with a specific token limit.
# This makes the tests more logical and avoids the MAX_TOKENS error.
TEST_CASES = [
    {
        "key": "summary_prompt",
        "prompt": "In one or two sentences, what is the eligibility process?",
        "tokens": 50
    },
    {
        "key": "paragraph_prompt",
        "prompt": "Briefly explain the key factors of the eligibility process for a family.",
        "tokens": 150
    },
    {
        "key": "detailed_prompt",
        "prompt": "Please provide a detailed and comprehensive explanation of the entire eligibility process for an applicant family, including the key factors that a Public Housing Authority must consider according to the guidebook.",
        "tokens": 300
    }
]


def run_grid_search():
    """
    Runs the full grid search and prints the results.
    """
    all_results = []

    # --- Loop through all combinations ---
    for model_name in MODELS_TO_TEST:
        for test_case in TEST_CASES:
            prompt_key = test_case["key"]
            prompt_text = test_case["prompt"]
            token_count = test_case["tokens"]

            print(f"--- Testing: [Model: {model_name}] | [Prompt: {prompt_key}] | [Tokens: {token_count}] ---")

            # Create the request payload
            payload = {
                "model_name": model_name,
                "prompt": prompt_text,
                "max_output_tokens": token_count
            }

            try:
                response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload), timeout=120)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

                result = response.json()
                
                if "error" in result:
                    print(f"    -> FAILED: API returned an error: {result['error']}\n")
                    # Append the error from the JSON body
                    all_results.append({
                        "model_name": model_name,
                        "prompt_key": prompt_key,
                        "requested_tokens": token_count,
                        "error": result['error']
                    })
                else:
                    result['prompt_key'] = prompt_key # Add this for easier analysis
                    all_results.append(result)
                    print(f"    -> Success! TTFT: {result['time_to_first_token']:.2f}s, Total Time: {result['total_time']:.2f}s\n")

            except requests.exceptions.RequestException as e:
                print(f"    -> FAILED: {e}\n")
                all_results.append({
                    "model_name": model_name,
                    "prompt_key": prompt_key,
                    "requested_tokens": token_count,
                    "error": str(e)
                })

            time.sleep(10)  # Increased sleep time slightly to be safe

    # --- Print Final Summary Table ---
    print("\n\n========================= GRID SEARCH SUMMARY =========================")
    # Header
    print(f"{'Model':<22} | {'Prompt':<18} | {'Req Tokens':<10} | {'Actual Tokens':<13} | {'TTFT (s)':<10} | {'Total Time (s)':<14}")
    print("-" * 105)

    # Rows
    for res in all_results:
        if "error" in res:
            print(f"{res.get('model_name', 'N/A'):<22} | {res.get('prompt_key', 'N/A'):<18} | {res.get('requested_tokens', 'N/A'):<10} | {'ERROR':<13} | {'-':<10} | {'-':<14}")
        else:
            # Use .get() for safer access to all keys
            model = res.get('model_name', 'N/A')
            prompt = res.get('prompt_key', 'N/A')
            req_tokens = res.get('requested_tokens', 'N/A')
            comp_tokens = res.get('completion_tokens', 'N/A')
            ttft = res.get('time_to_first_token', 0)
            total_time = res.get('total_time', 0)
            print(f"{model:<22} | {prompt:<18} | {req_tokens:<10} | {comp_tokens:<13} | {ttft:<10.2f} | {total_time:<14.2f}")

    print("=========================================================================================\n")


if __name__ == "__main__":
    run_grid_search()