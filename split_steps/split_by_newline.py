import json
import re

with open("../datasets/organised_asprm_m_training_data_2.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f]

with open("../datasets/split_by_newline_asprm_m_training_data.jsonl", "w") as f:
    for item in dataset:
        question = item["question"].strip()
        solution = item["solution"].strip()
        marker = "\u043a\u0438"
        query = question + " " + re.sub(r' +', ' ', re.sub(r'(\n+)', r'\1 ' + marker + ' ', solution)).strip()
        if query [-2:-1] != marker:
            query += " " + marker
        new_item = {
            "id": item["id"],
            "query": query,
            "question": question,
            "solution": solution,
            "is_answer_correct": item["is_answer_correct"],
            "gold_answer": item["gold_answer"],
            "final_answer": item["final_answer"]
        }
        f.write(json.dumps(new_item) + "\n")

