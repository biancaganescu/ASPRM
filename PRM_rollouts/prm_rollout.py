import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams

STEP_SEPARATOR = "ки"
CORRECT_LABEL = "к"
INCORRECT_LABEL = "и"

 
@dataclass
class Entry:
    query: str          # question + response with ки markers
    question: str       # original question
    solution: str       # original solution
    gold_answer: str       # the correct final answer
    index: int              # position in the dataset
    step_prefixes: list[str] = field(default_factory=list)
    num_steps: int = 0
    final_answer: str = None   # the model's final answer (after all steps)

def extract_response_answer(response):
    if "The answer is:" in response:
        return response.split("The answer is:")[1].strip()
    return None

def normalise_answer(answer):
    if answer is None:
        return None
    answer = answer.strip().lower()
    answer = answer.replace(",", "").replace("$", "").replace("%", "")
    answer = answer.rstrip(".")
    try:
        return str(float(answer))
    except ValueError:
        return answer

def check_answer(gold_answer, model_answer):
    if gold_answer is None or model_answer is None:
        return False
    return normalise_answer(gold_answer) == normalise_answer(extract_response_answer(model_answer))

def load_dataset(file_path):
    entries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
                row = json.loads(line)
    
                entry = Entry(
                    question=row["question"],
                    solution=row["solution"],
                    query=row["query"],
                    gold_answer=row["gold_answer"],
                    final_answer=row["final_answer"],
                    index=i,
                )
                entries.append(entry)
 
    print(f"Loaded {len(entries)} entries")
    return entries

def build_prefixes(entries):
    all_prefixes = []

    for idx, entry in enumerate(entries):
        question = entry.question.strip()
        steps = entry.solution.split(STEP_SEPARATOR)
        entry.num_steps = len(steps) - 1  # number of steps is one less than number of segments

        if entry.num_steps == 0:
            print(f"[WARN] Sample {idx}: no step markers found")
            continue
        
        prefixes = []
        for step_idx in range(entry.num_steps):
            prefix = question + " " + " ".join(steps[:step_idx + 1]).strip()
            # print("Prefix for step", step_idx, ":", prefix)
            prefixes.append(prefix)
        
        entry.step_prefixes = prefixes
        for step_idx, prefix in enumerate(prefixes):
            all_prefixes.append((entry.index, step_idx, prefix))
    print(f"Built prefixes for {len(entries)} entries, total prefixes: {len(all_prefixes)}")
    return all_prefixes

def run_rollouts(llm, all_prefixes, sampling_params, batch_size=256):
    results = {}
    total = len(all_prefixes)
    for start in tqdm(range(0, total, batch_size), desc="Running rollouts"):
        end = min(start + batch_size, total)
        batch = all_prefixes[start:end]

        instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"
        response_start="\n\n### Response:"
        prompts = [instruction + prefix + response_start for _, _, prefix in batch]
        responses = llm.generate(prompts, sampling_params)

        for (entry_idx, step_idx, prefix), response in zip(batch, responses):
            outputs = [output.text for output in response.outputs]
            results[(entry_idx, step_idx)] = outputs
    return results

def evaluate_rollouts(entries, rollout_results):
    labelled = []
    for entry in entries:
        if entry.num_steps == 0:
            continue

        steps = entry.solution.split(STEP_SEPARATOR)
        labelled_response = entry.question.strip() + " "  # start with the question
        step_label_list = [] 
        label_list = []  

        for step_idx in range(entry.num_steps):
            key = (entry.index, step_idx)
            rollout_outputs = rollout_results.get(key, [])
 
            any_correct = any(check_answer(entry.gold_answer, output) for output in rollout_outputs)
 
            label = CORRECT_LABEL if any_correct else INCORRECT_LABEL
            binary_label = 1 if any_correct else 0
 
            labelled_response += steps[step_idx] + label
 
            step_text = steps[step_idx]
            step_label_list.append(step_text)
            step_label_list.append("1" if any_correct else "0")
            label_list.append(binary_label)
 
        labelled.append({
            "query": entry.query,
            "labelled_response": labelled_response,
            "step_label_list": step_label_list,
            "label_list": label_list,
        })

    return labelled

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="PRM Rollout Labeler (AdaptiveStep format)")
    parser.add_argument("--model", type=str, default="Lux0926/metamath_mistral_7b", help="Model name or path")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset name")
    parser.add_argument("--output", type=str, default="./labeled_output.jsonl", help="Output JSONL path")
    parser.add_argument("--rollouts", type=int, default=8, help="Number of rollouts per step")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens per rollout")
    parser.add_argument("--batch_size", type=int, default=5000, help="Prefix batch size for generation")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--gpu_memory", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--max_examples", type=int, default=None, help="Only use the first N examples")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if args.max_examples is not None:
        dataset = dataset[:args.max_examples]
        print(f"Using first {len(dataset)} examples")

    all_prefixes = build_prefixes(dataset)

    print(f"\nLoading model: {args.model} (tp={args.tp})")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.rollouts,
    )

    print(f"\nGenerating {args.rollouts} rollouts per step ...")
    rollout_results = run_rollouts(llm, all_prefixes, sampling_params, args.batch_size)

    print("\nLabeling steps ...")
    labeled_data = evaluate_rollouts(dataset, rollout_results)

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in labeled_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(labeled_data)} labeled samples to {output_path}")
 
 

