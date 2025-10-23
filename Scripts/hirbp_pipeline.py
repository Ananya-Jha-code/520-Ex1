import os
import json
import re
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
MAX_ITER = 2  # Maximum refinement iterations

# Google Drive folder for saving results
DRIVE_FOLDER = "/content/drive/MyDrive/human_eval_hirbp"
os.makedirs(DRIVE_FOLDER, exist_ok=True)

# -----------------------------
# Load HumanEval dataset
# -----------------------------
human_eval = load_dataset("openai_humaneval")['test']

# -----------------------------
# Prompt Templates for HIRBP Roles
# -----------------------------
def developer_prompt(problem_text):
    return f"""
You are a Python developer. Solve the following problem step-by-step:

Problem:
{problem_text}

Instructions:
- Think step by step.
- Write the function get_solution.
- Include 'from typing import List' if needed.
- Mark start and end keywords.

Start get_solution
...
End get_solution
"""

def tester_prompt(problem_text, code):
    return f"""
You are a Tester. Evaluate the following function against the problem:

Problem:
{problem_text}

Function:
{code}

1. Run it on sample inputs.
2. If it fails, summarize the reason briefly.
3. If it passes, just confirm success.
"""

def debugger_prompt(problem_text, code, failure_summary):
    return f"""
You are a Debugger. The following code failed:

Problem:
{problem_text}

Function:
{code}

Failure Summary:
{failure_summary}

Instructions:
- Fix errors, add missing imports if needed.
- Handle edge cases and off-by-one issues.
- Return only the corrected get_solution function.
- Mark start and end keywords.
"""

# -----------------------------
# Code extraction utility
# -----------------------------
def extract_code(text):
    if not text:
        return ""
    text = re.sub(r"```(?:python)?\n", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

# -----------------------------
# Gemini & GPT code generation
# -----------------------------
def generate_code_gemini(prompt):
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    if hasattr(response, "text") and response.text:
        return extract_code(response.text)
    elif hasattr(response, "candidates") and response.candidates:
        return extract_code(response.candidates[0].content.parts[0].text)
    return ""

def generate_code_gpt(prompt, client):
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0
    )
    return extract_code(response.choices[0].message.content)

def generate_code_with_model(model_name, prompt, gpt_client=None):
    if model_name.lower().startswith("gpt"):
        return generate_code_gpt(prompt, gpt_client)
    else:
        return generate_code_gemini(prompt)

# -----------------------------
# Hybrid Iterative Refinement Pipeline
# -----------------------------
def run_hirbp(models=["GPT-4", "Gemini"], problems_list=None, max_iterations=MAX_ITER):
    """
    Run Hybrid Prompt-Refinement (HIRBP) pipeline.
    Developer -> Tester -> Debugger loop for max_iterations
    """
    if problems_list is None:
        problems_list = human_eval.select([0, 3, 6, 8])  # Known failure cases

    gpt_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    hirbp_results = []

    for problem in tqdm(problems_list):
        problem_id = problem["task_id"]
        problem_text = problem["prompt"]

        for model_name in models:
            iteration = 0
            success = False
            current_code = developer_prompt(problem_text)  # Initial prompt
            generated_code = ""

            while iteration < max_iterations and not success:
                # Step 1: Developer generates code
                generated_code = generate_code_with_model(model_name, current_code, gpt_client)

                # Step 2: Tester evaluates
                score = evaluate_code(problem, generated_code)
                if score.get("pass@1", 0) == 1:
                    success = True
                    break

                # Step 3: Tester feedback summary
                failure_summary = "Function failed test cases"  # Simplified summary

                # Step 4: Debugger refines code
                current_code = debugger_prompt(problem_text, generated_code, failure_summary)
                iteration += 1

            hirbp_results.append({
                "problem_id": problem_id,
                "model": model_name,
                "iterations": iteration,
                "final_code": generated_code,
                "score": score
            })

    # Save results
    hirbp_jsonl_path = os.path.join(DRIVE_FOLDER, "hirbp_results.jsonl")
    with open(hirbp_jsonl_path, "w") as f:
        for entry in hirbp_results:
            f.write(json.dumps(entry) + "\n")
    print(f"HIRBP results saved to: {hirbp_jsonl_path}")

    return hirbp_results
