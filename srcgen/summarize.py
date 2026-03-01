import json
import time
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

BATCH = 20

client = OpenAI(
    api_key='REMOVED',
    base_url='http://example.com:8317/v1/'
)
model = 'gemini-3-flash-preview'

# client = OpenAI(
#     base_url='https://openrouter.ai/api/v1',
#     api_key='REMOVED'
# )
# model = 'tngtech/deepseek-r1t2-chimera:free'


def format_path(c):
    return f"summary/{c['novelId']}_{c['chapterId']}.txt"


def summarize_batch(batch):
    chapters_text = "\n\n".join(
        f'=== 章节 [{i}] ===\n{c["content"][:8000]}' for i, c in enumerate(batch)
    )
    prompt = (
        f"以下有 {len(batch)} 个章节，用 === 章节 [编号] === 分隔。"
        "请分别概括每章约 500 字的主要情节，不要偷工减料。"
        "严格按以下 JSON 格式输出，JSON 里用 Markdown 格式，不要输出任何其他内容，始终使用简体中文:\n"
        '[{"id": 0, "summary": "..."}, {"id": 1, "summary": "..."}, ...]\n\n'
        + chapters_text
    )
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="high",
    )
    print(r.usage)
    text = r.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    text = text.strip('`}"\n ')
    try:
        summaries = json.loads(text)
    except json.JSONDecodeError as e:
        p = f'fail/{int(time.time())}.txt'
        print("Failed to parse JSON. Saving to", p)
        with open(p, "w") as f:
            f.write(text)
        raise e
    for item in summaries:
        c = batch[item["id"]]
        with open(format_path(c), "w") as f:
            f.write(item["summary"])
        print(f"  ✓ {format_path(c)}")
    print(f"Sample: {summaries[0]['summary']}")

with open("chapters_sample.json", "r") as f:
    chapters = json.load(f)

tasks = [c for c in chapters if "content" in c and not os.path.exists(format_path(c)) and len(c["content"].strip()) > 1000]
batches = [tasks[i : i + BATCH] for i in range(0, len(tasks), BATCH)]
print(f"Total: {len(tasks)} chapters in {len(batches)} batches")

with ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(summarize_batch, b) for b in batches]
    for f in tqdm(as_completed(futures), total=len(futures)):
        try:
            f.result()
        except Exception as e:
            print(f"Error: {e}")
