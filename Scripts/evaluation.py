import re
from evaluate import load

# Initialize code evaluation metric
code_eval_metric = load("code_eval")

def extract_get_solution_code(text: str) -> str:
    """
    Extract the function code between start and end keywords.
    """
    if not text:
        return ""
    lines = text.splitlines()
    start_idx, end_idx = None, None

    for i, line in enumerate(lines):
        if 'start' in line.strip().lower() and start_idx is None:
            start_idx = i + 1
        elif 'end' in line.strip().lower() and start_idx is not None:
            end_idx = i
            break

    if start_idx is not None and end_idx is not None and start_idx < end_idx:
        return "\n".join(lines[start_idx:end_idx]).strip()
    return text.strip()

def evaluate_code(problem, generated_code):
    """
    Evaluate a single code snippet against a HumanEval problem.
    Returns score dictionary with pass@1 or error.
    """
    clean_code = extract_get_solution_code(generated_code)

    try:
        score = code_eval_metric.compute(predictions=[clean_code], references=[problem])
    except Exception as e:
        score = {"pass@1": 0, "error": str(e)}

    return score

def check_pass_local(problem, code):
    """
    Quick local pass/fail check using exec (safe for small HumanEval tests)
    """
    try:
        local_vars = {}
        exec(code, {}, local_vars)
        exec(problem["test"], {}, local_vars)
        return 1
    except Exception:
        return 0
