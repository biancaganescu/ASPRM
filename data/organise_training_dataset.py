import json
import re
from datasets import load_dataset

def common_prefix(s1, s2):
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return s1[:i]

def trim_to_sentence(text):
    last = max(text.rfind(". "), text.rfind(".$"), text.rfind(".\n"), text.rfind("?"), text.rfind("?\n"))
    if last == -1:
        return text
    return text[:last + 1]

# Load gold answers
gsm8k = load_dataset("gsm8k", "main", split="train")
answer_lookup = {}
for row in gsm8k:
    q = row["question"].strip()
    a = row["answer"].split("####")[-1].strip()
    answer_lookup[q] = a

deepmath = load_dataset("zwhe99/DeepMath-103K", split="train")
for row in deepmath:
    q = row["question"].strip()
    a = row["final_answer"].strip()
    answer_lookup[q] = a

def extract_boxed(text):
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None

configs = ['algebra', 'counting_and_probability', 'geometry',
           'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
for config in configs:
    math_ds = load_dataset("EleutherAI/hendrycks_math", config, split="train")
    for row in math_ds:
        q = row["problem"].strip()
        a = extract_boxed(row["solution"])
        if a is not None:
            answer_lookup[q] = a

# Load entries
with open("ASPRM_M_Training_Data.jsonl") as f:
    entries = [json.loads(line) for line in f]

# Step 1: Group consecutive entries by detecting prefix drops
groups = [[entries[0]]]
for entry in entries[1:]:
    prefix = common_prefix(groups[-1][0]["query"], entry["query"])
    if len(prefix) > 20:
        groups[-1].append(entry)
    else:
        groups.append([entry])

# Step 2: Process each group
results = []
no_answer_groups = []
i = 0
from_group = 0
from_lookup = 0
no_answer = 0

for group in groups:
    question = group[0]["query"]
    for entry in group[1:]:
        question = common_prefix(question, entry["query"])
    question = trim_to_sentence(question)

    # Try 1: extract gold answer from a correct entry in the group
    gold_answer = None
    for entry in group:
        if entry["response"].strip()[-1] == "\u043a":
            sol = entry["query"][len(question):].lstrip()
            sol_clean = re.sub(r' +', ' ', sol.replace("\u043a\u0438", " "))
            if "The answer is: " in sol_clean:
                gold_answer = sol_clean[sol_clean.find("The answer is: ") + len("The answer is: "):].strip()
                from_group += 1
                break

    # Try 2: fallback to lookup
    if gold_answer is None:
        for q, a in answer_lookup.items():
            if question.startswith(q) or q.startswith(question):
                gold_answer = a
                from_lookup += 1
                break

    if gold_answer is None:
        no_answer += 1
        no_answer_groups.append({
            "question": question,
            "num_entries": len(group),
            "has_correct": any(e["response"].strip()[-1] == "\u043a" for e in group),
            "first_query": group[0]["query"][:300],
        })
        continue

    for entry in group:
        query = entry["query"]
        response = entry["response"]
        solution = query[len(question):].lstrip()
        reformatted_solution = re.sub(r' +', ' ', solution.replace("\u043a\u0438", " "))
        response_without_question = response[len(question):].lstrip()
        solution = solution.replace("\u043a\u0438", "").strip()
        solution = re.sub(r' +', ' ', solution).strip()
        results.append({
            "id": i,
            # "query": query,
            # "response": response,
            "question": question.strip(),
            "gold_answer": gold_answer,
            "solution": solution,
            # "reformatted_solution": reformatted_solution,
            # "response_without_question": response_without_question,
            "is_answer_correct": response.strip()[-1] == "\u043a",
            "final_answer": reformatted_solution[reformatted_solution.find("The answer is: ") + len("The answer is: "):].strip() if "The answer is:" in reformatted_solution else None,
        })
        i += 1

print(f"Gold answer from group: {from_group}")
print(f"Gold answer from lookup: {from_lookup}")
print(f"No gold answer: {no_answer}")
print(f"Total groups: {len(groups)}, Total entries: {len(results)}")

if no_answer_groups:
    print(f"\n{len(no_answer_groups)} groups with no gold answer:\n")
    for g in no_answer_groups:
        print(f"  Q: {g['question']}")
        print(f"  Entries: {g['num_entries']}, Has correct: {g['has_correct']}")
        print()

with open("organised_asprm_m_training_data_2.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")