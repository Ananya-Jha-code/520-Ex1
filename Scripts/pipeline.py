import os
import json
from tqdm import tqdm
from datasets import load_dataset
import openai
import google.generativeai as genai
from evaluation import evaluate_code, extract_get_solution_code

# -----------------------------
# Config
# -----------------------------
GPT_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"

# Google Drive folder for saving results
DRIVE_FOLDER = "/content/drive/MyDrive/human_eval_results"
os.makedirs(DRIVE_FOLDER, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
human_eval = load_dataset("openai_humaneval")['test']

# -----------------------------
# Prompt strategies
# -----------------------------
def chain_of_thought(problem_text):
    return f"""
You are a skilled Python developer.
{problem_text}
Think step by step about the logic before writing the code.
Then write the final, correct function implementation.
Return the code in a function named get_solution
Mark the start and end of get_solution with start and end keywords
"""

def stepwise_chain_of_thought(problem_text):
    return f"""
You are to solve the following problem step-by-step.

1. Break down the logic clearly in small numbered steps.
2. Then write the Python function.
3. Double-check for correctness and edge cases.
4. Return the code in a function named get_solution
5. Mark the start and end of get_solution with start and end keywords

Problem:
{problem_text}
"""

prompt_strategies = [("CoT", chain_of_thought), ("SCoT", stepwise_chain_of_thought)]

# -----------------------------
# Extract clean code helper
# -----------------------------
def extract_code(text):
    if not text:
        return ""
    text = re.sub(r"```(?:python)?\n", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

# -----------------------------
# Gemini generation
# -----------------------------
def generate_code_gemini(prompt):
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    if hasattr(response, "text") and response.text:
        return extract_code(response.text)
    elif hasattr(response, "candidates") and response.candidates:
        return extract_code(response.candidates[0].content.parts[0].text)
    return ""

# -----------------------------
# GPT generation
# -----------------------------
def generate_code_gpt(prompt, client):
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0
    )
    return extract_code(response.choices[0].message.content)

# -----------------------------
# Unified generation helper
# -----------------------------
def generate_code_with_model(model_name, prompt, gpt_client=None):
    if model_name.lower().startswith("gpt"):
        return generate_code_gpt(prompt, gpt_client)
    else:
        return generate_code_gemini(prompt)

# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(models=["GPT-4", "Gemini"], num_problems=10):
    generated_results = []

    # Initialize OpenAI client
    gpt_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    # Select subset of problems
    selected_problems = human_eval.select(range(num_problems))

    for sample in tqdm(selected_problems):
        problem_id = sample['task_id']

        for strat_name, strat_fn in prompt_strategies:
            prompt = strat_fn(sample['prompt'])

            for model_name in models:
                code = generate_code_with_model(model_name, prompt, gpt_client)
                generated_results.append({
                    "problem_id": problem_id,
                    "model": model_name,
                    "strategy": strat_name,
                    "prompt": prompt,
                    "generated_code": code,
                    "test_cases": sample["test"]
                })

    # Save generated results
    jsonl_path = os.path.join(DRIVE_FOLDER, "generated_code_prompts.jsonl")
    with open(jsonl_path, "w") as f:
        for entry in generated_results:
            f.write(json.dumps(entry) + "\n")
    print(f"Generated code saved to: {jsonl_path}")

    return generated_results, selected_problems

# -----------------------------
# Evaluate all results
# -----------------------------
def evaluate_pipeline_results(generated_results, selected_problems):
    evaluation_results = []

    for entry in tqdm(generated_results):
        try:
            problem = next(p for p in selected_problems if p["task_id"] == entry["problem_id"])
            score = evaluate_code(problem, entry["generated_code"])
        except Exception as e:
            score = {"pass@1": 0, "error": str(e)}

        evaluation_results.append({
            "problem_id": entry["problem_id"],
            "model": entry["model"],
            "strategy": entry["strategy"],
            "generated_code": entry["generated_code"],
            "score": score
        })

    # Save evaluation results
    eval_jsonl_path = os.path.join(DRIVE_FOLDER, "evaluation_results.jsonl")
    with open(eval_jsonl_path, "w") as f:
        for entry in evaluation_results:
            f.write(json.dumps(entry) + "\n")
    print(f"Evaluation results saved to: {eval_jsonl_path}")

    return evaluation_results
