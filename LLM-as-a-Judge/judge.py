# from tqdm import tqdm
# import json
# import re
# import requests

# # prompt = "I will provide a math problem along with a solution. They will be formatted as\
# # follows:\n\n\
# # [Math Problem]\n\n\
# # <math_problem>\n\
# # ...(math problem)...\n\
# # </math_problem>\n\n\
# # [Solution]\n\n\
# # <paragraph_1>\n\
# # ...(paragraph 1 of solution)...\n\
# # </paragraph_1>\n\n\
# # ...\n\n\
# # <paragraph_n>\n\
# # ...(paragraph n of solution)...\n\
# # </paragraph_n>\n\n\
# # Your task is to review each paragraph of the solution in sequence, analyzing,\
# # verifying, and critiquing the reasoning in detail. You need to provide the\
# # analyses and the conclusion in the following format:\n\n\
# # <analysis_1>\n\
# # ...(analysis of paragraph 1)...\n\
# # </analysis_1>\n\n\
# # ...\n\n\
# # <analysis_n>\n\
# # ...(analysis of paragraph n)...\n\
# # </analysis_n>\n\n\
# # <conclusion>\n\
# # Correct/Incorrect\n\
# # </conclusion>\n\n\
# # * When you analyze each paragraph, you should use proper verification,\
# # recalculation, or reflection to indicate whether it is logically and\
# # mathematically valid. Please elaborate on the analysis process carefully.\n\n\
# # * If an error is detected in any paragraph, you should describe the nature and\
# # cause of the error in detail, and suggest how to correct the error or the correct\
# # approach. Once a paragraph is found to contain any error, stop further analysis\
# # of subsequent paragraphs (as they may depend on the identified error) and directly\
# # provide the conclusion of \"Incorrect\".\n\n\
# # For instance, given a solution of five paragraphs, if an error is found in the\
# # third paragraph, you should reply in the following format:\n\n\
# # <analysis_1>\n\
# # ...(analysis of paragraph 1)...\n\
# # </analysis_1>\n\n\
# # <analysis_2>\n\
# # ...(analysis of paragraph 2)...\n\n\
# # </analysis_2>\n\n\
# # </analysis_3>\n\
# # <analysis_3>\n\
# # ...(analysis of paragraph 3; since an error is found here, also provide detailed\
# # critique and correction guideline)...\n\
# # </analysis_3>\n\n\
# # <conclusion>\n\
# # Incorrect\n\
# # </conclusion>\n\n\
# # Note that the analyses of paragraphs 4 and 5 should be skipped as the paragraph\
# # 3 has been found to contain an error.\n\n\
# # * Respond with your analyses and conclusion directly.\n\n\
# # --------------------------------------------------\n\n\
# # The following is the math problem and the solution for you task:\n\n\
# # [Math Problem]\n\n"

# # solution_prompt = "\n\n[Solution]\n\n"

# instruction_prompt = """\
# I will provide a math problem along with a solution. They will be formatted as follows:

# [Math Problem]

# <math_problem>
# ...(math problem)...
# </math_problem>

# [Solution]

# <paragraph_1>
# ...(paragraph 1 of solution)...
# </paragraph_1>

# ...

# <paragraph_n>
# ...(paragraph n of solution)...
# </paragraph_n>

# Your task is to review each paragraph of the solution in sequence, analyzing, verifying, and critiquing the reasoning in detail. You need to provide the analyses and the conclusion in the following format:

# <analysis_1>
# ...(analysis of paragraph 1)...
# </analysis_1>

# ...

# <analysis_n>
# ...(analysis of paragraph n)...
# </analysis_n>

# <conclusion>
# Correct/Incorrect
# </conclusion>

# * When you analyze each paragraph, you should use proper verification, recalculation, or reflection to indicate whether it is logically and mathematically valid. Please elaborate on the analysis process carefully.

# * If an error is detected in any paragraph, you should describe the nature and cause of the error in detail, and suggest how to correct the error or the correct approach. Once a paragraph is found to contain any error, stop further analysis of subsequent paragraphs (as they may depend on the identified error) and directly provide the conclusion of "Incorrect."

# For instance, given a solution of five paragraphs, if an error is found in the third paragraph, you should reply in the following format:

# <analysis_1>
# ...(analysis of paragraph 1)...
# </analysis_1>

# <analysis_2>
# ...(analysis of paragraph 2)...
# </analysis_2>

# <analysis_3>
# ...(analysis of paragraph 3; since an error is found here, also provide detailed
# critique and correction guideline)...
# </analysis_3>

# <conclusion>
# Incorrect
# </conclusion>

# Note that the analyses of paragraphs 4 and 5 should be skipped as the paragraph 3 has been found to contain an error.

# * Respond with your analyses and conclusion directly.

# --------------------------------------------------
# The following is the math problem and the solution for you task:

# [Math Problem]

# """
# solution_prompt = "\n\n[Solution]\n\n"


# with open("../PRM_rollouts/datasets/split_by_newline_asprm_m_training_data.jsonl", "r") as f:
#     dataset = [json.loads(line) for line in f]

# with open("../PRM_rollouts/datasets/judge_evaluation_data_split_by_newline.jsonl", "w") as f:
#     for item in tqdm(dataset[:10000]):
#         question = item["question"].strip()
#         solution = item["solution"].strip()
        
#         parts = re.split(r"(\n+)", solution)
#         steps = []
#         for i in range(0, len(parts) - 1, 2):
#             steps.append(parts[i] + parts[i + 1])
#         if len(parts) % 2 == 1:
#             steps.append(parts[-1]) 

#         for i, step in enumerate(steps):
#             step = step.strip()
#             if step:
#                 step = f"<paragraph_{i+1}>\n" + step + f"\n</paragraph_{i+1}>"
        
#         solution = "\n\n".join(steps)

#         llm_input = instruction_prompt + question + solution_prompt + solution

#         response = requests.post(
#         url="https://openrouter.ai/api/v1/chat/completions",
#         headers={
#             "Authorization": "Bearer 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",
#         },
#         data=json.dumps({
#             "model": "qwen/qwen-2.5-72b-instruct",
#             "messages": [
#             {
#                 "role": "user",
#                 "content": llm_input
#             }
#         ],
#         "temperature": 0})
#         )

#         if "choices" not in response.json():
#             print("Error in comparison response:", response.json())
#             total -= 1
#             continue
#         else:
#             try:
#                 f.write(json.dumps({
#                     "id": item["id"],
#                     "query": item["query"],
#                     "question": item["question"],
#                     "solution": item["solution"],
#                     "llm_output": response.json()["choices"][0]["message"]["content"]
#                 }) + "\n")
#             except Exception as e:
#                 print("Error processing response:", e)


import asyncio
import aiohttp
import json
import re
from tqdm.asyncio import tqdm_asyncio

CONCURRENCY = 100
OPENROUTER_API_KEY = ""
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
INPUT_PATH = "../PRM_rollouts/datasets/split_by_newline_asprm_m_training_data.jsonl"
OUTPUT_PATH = "../PRM_rollouts/datasets/simplified_judge_evaluation_data_split_by_newline.jsonl"
MAX_ITEMS = 1000

instruction_prompt = """\
I will provide a math problem along with a solution. They will be formatted as follows:

[Math Problem]

<math_problem>
...(math problem)...
</math_problem>

[Solution]

<paragraph_1>
...(paragraph 1 of solution)...
</paragraph_1>

...

<paragraph_n>
...(paragraph n of solution)...
</paragraph_n>

Your task is to review each paragraph of the solution in sequence, analyzing, verifying, and critiquing the reasoning in detail. You need to provide the analyses and the conclusion in the following format:

<analysis_1>
...(analysis of paragraph 1)...
</analysis_1>

...

<analysis_n>
...(analysis of paragraph n)...
</analysis_n>

<conclusion>
Correct/Incorrect
</conclusion>

* When you analyze each paragraph, you should use proper verification, recalculation, or reflection to indicate whether it is logically and mathematically valid. Please elaborate on the analysis process carefully.

* If an error is detected in any paragraph, you should describe the nature and cause of the error in detail, and suggest how to correct the error or the correct approach. Once a paragraph is found to contain any error, stop further analysis of subsequent paragraphs (as they may depend on the identified error) and directly provide the conclusion of "Incorrect."

For instance, given a solution of five paragraphs, if an error is found in the third paragraph, you should reply in the following format:

<analysis_1>
...(analysis of paragraph 1)...
</analysis_1>

<analysis_2>
...(analysis of paragraph 2)...
</analysis_2>

<analysis_3>
...(analysis of paragraph 3; since an error is found here, also provide detailed
critique and correction guideline)...
</analysis_3>

<conclusion>
Incorrect
</conclusion>

Note that the analyses of paragraphs 4 and 5 should be skipped as the paragraph 3 has been found to contain an error.

* Respond with your analyses and conclusion directly.

--------------------------------------------------
The following is the math problem and the solution for you task:

[Math Problem]

"""
instruction_prompt = """
I will provide a math problem along with a solution. They will be formatted as follows:

[Math Problem]

Step 1: ...(step 1 of solution)...

Step 2: ...(step 2 of solution)...

...

Step n: ...(step n of solution)...

Your task is to review each paragraph of the solution in sequence, analyzing, verifying, and critiquing the reasoning in detail. You need to only provide the conclusion in the following format:

* If an error is detected in any paragraph, you should describe the nature and cause of the error in detail, and suggest how to correct the error or the correct approach.

You should reply in the following format:

Step 1: <analysis> Therefore, step 1 is [correct/incorrect]

Step 2: <analysis> Therefore, step 2 is [correct/incorrect]

...

Step n: <analysis> Therefore, step n is [correct/incorrect]

The final answer is [correct/incorrect].

* When you analyze each paragraph, you should use proper verification, recalculation, or reflection to indicate whether it is logically and mathematically valid. Please elaborate on the analysis process carefully.

* Respond with your analysis and conclusion for each step directly.

--------------------------------------------------
The following is the math problem and the solution for you task:

[Math Problem]

"""

solution_prompt = "\n\n[Solution]\n\n"


def format_solution(solution: str) -> str:
    """Split solution by newlines and wrap each step in paragraph tags."""
    parts = re.split(r"(\n+)", solution.strip())
    steps = []
    for i in range(0, len(parts) - 1, 2):
        steps.append(parts[i] + parts[i + 1])
    if len(parts) % 2 == 1:
        steps.append(parts[-1])

    formatted = []
    para_idx = 1
    for step in steps:
        stripped = step.strip()
        if stripped:
            formatted.append(f"Step {para_idx}: {stripped}")
            para_idx += 1

    return "\n\n".join(formatted)


async def call_openrouter(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    item: dict,
    max_retries: int = 3,
) -> dict | None:
    """Send one item to OpenRouter with retries and concurrency limiting."""
    question = item["question"].strip()
    solution = format_solution(item["solution"])
    llm_input = instruction_prompt + question + solution_prompt + solution

    payload = {
        "model": "qwen/qwen-2.5-72b-instruct",
        "messages": [{"role": "user", "content": llm_input}],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            async with semaphore:
                async with session.post(
                    OPENROUTER_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    data = await resp.json()
                    if data is None:
                        print(f"Empty response for item {item['id']}, retrying...")
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    # Rate limited — back off and retry
                    if resp.status == 429:
                        wait = 2 ** (attempt + 1)
                        print(f"Rate limited on item {item['id']}, retrying in {wait}s...")
                        await asyncio.sleep(wait)
                        continue

                    if "choices" not in data:
                        print(f"Unexpected response for item {item['id']}: {data}")
                        return None

                    return {
                        "id": item["id"],
                        "query": item["query"],
                        "question": item["question"],
                        "solution": item["solution"],
                        "llm_output": data["choices"][0]["message"]["content"],
                    }

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            wait = 2 ** (attempt + 1)
            print(f"Error on item {item['id']} (attempt {attempt+1}): {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)

    print(f"Failed after {max_retries} retries for item {item['id']}")
    return None


async def main():
    with open(INPUT_PATH, "r") as f:
        dataset = [json.loads(line) for line in f]

    items = dataset[:MAX_ITEMS]
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        tasks = [call_openrouter(session, semaphore, item) for item in items]
        results = await tqdm_asyncio.gather(*tasks, desc="Processing")

    successes = 0
    failures = 0
    with open(OUTPUT_PATH, "w") as f:
        for result in results:
            if result is not None:
                f.write(json.dumps(result) + "\n")
                successes += 1
            else:
                failures += 1

    print(f"\nDone: {successes} succeeded, {failures} failed out of {len(items)} total.")


if __name__ == "__main__":
    asyncio.run(main())