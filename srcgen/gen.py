import json
import time
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import os
import random
import argparse
from collections import deque
# import httpx

from metrics import tokens_counter, chars_counter

# custom_headers = {
#     "User-Agent": "Cline/3.57.1",
#     "X-Stainless-Retry-Count": "0",
#     "X-Stainless-Lang": "js",
#     "X-Stainless-Package-Version": "6.18.0",
#     "X-Stainless-OS": "Windows",
#     "X-Stainless-Arch": "x64",
#     "X-Stainless-Runtime": "node",
#     "X-Stainless-Runtime-Version": "v22.21.1",
# }
# custom_http_client = httpx.Client(headers=custom_headers)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen3-coder-plus")
parser.add_argument("--batch", type=int, default=20)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--worker", type=int, default=5)
args = parser.parse_args()

random.seed(42)

BATCH = args.batch
model = args.model

if args.model == 'custom':
    from model import MyOpenAI
    client = MyOpenAI()
else:
    if '/' not in args.model:
        # client = OpenAI(
        #     api_key='REMOVED',
        #     base_url='https://ark.cn-beijing.volces.com/api/coding/v3',
        #     http_client=custom_http_client
        # )
        client = OpenAI(
            api_key='REMOVED',
            base_url='http://example.com:8317/v1/'
        )
        # client = OpenAI(
        #     api_key='REMOVED',
        #     base_url='https://opencode.ai/zen/v1/'
        # )
    else:
        client = OpenAI(
            api_key='REMOVED',
            base_url='https://openrouter.ai/api/v1/'
        )


def summary_path(c):
    return f"summary/{c['novelId']}_{c['chapterId']}.txt"


def output_path(c):
    return f"{args.output_dir}/{c['novelId']}_{c['chapterId']}.txt"


def update_cpm(stats, now, chars):
    stats["recent"].append((now, chars))
    while stats["recent"] and now - stats["recent"][0][0] > 60:
        stats["recent"].popleft()
    return sum(c for _, c in stats["recent"])


def generate_batch(batch):
    chapters_text = "\n\n".join(
        f"=== 章节 [{i}] ===\n{item['summary']}" for i, item in enumerate(batch)
    )
    prompt = (
        f"以下有 {len(batch)} 个章节概要，用 === 章节 [编号] === 分隔。"
        "请根据每个概要扩写成完整章节正文，保持连贯的叙事、细节和对话，每章节 2000 字左右，算准字数不要过长，算准字数不要过长！！！这些章节间不一定有剧情关联，不要混淆。"
        "严格按以下 JSON 格式输出，JSON 的 content 需要合理换行，不要输出任何其他内容，始终使用简体中文:\n"
        '[{"id": 0, "content": "..."}, {"id": 1, "content": "..."}, ...]\n\n'
        + chapters_text
    )
    
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are Cline, a helpful assistant."}, {"role": "user", "content": prompt}],
        # reasoning_effort="high",
        max_tokens=32768,
        stream=True,
        stream_options={"include_usage": True}
    )
    
    # use stream to get real-time stats
    text = ""
    last_cpm_update = 0.0
    pending_chars = 0
    for chunk in stream:
        if chunk.usage:
            print(chunk.usage)
            # update metrics
            tags = {
                "model": model,
                "batch_size": len(batch),
                "parallel": args.worker,
            }
            tokens_counter.add(chunk.usage.completion_tokens, {**tags, "type": "completion"})
            tokens_counter.add(chunk.usage.prompt_tokens, {**tags, "type": "prompt"})
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            text += content
            # update stats
            pending_chars += len(content)
            now = time.time()
            if now - last_cpm_update >= 0.2 and lock.acquire(False):
                try:
                    recent_chars = update_cpm(stats, now, pending_chars)
                    pbar.set_postfix(
                        CPM=f"{recent_chars}",
                    )
                finally:
                    lock.release()
                pending_chars = 0
                last_cpm_update = now
            chars_counter.add(len(content), {"parallel": args.worker, "model": model})
    with lock:
        update_cpm(stats, time.time(), pending_chars)
    
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    text = text.strip('`}"\n ')
    try:
        outputs = json.loads(text)
    except json.JSONDecodeError as e:
        p = f'fail_gen/{int(time.time())}.txt'
        print("Failed to parse JSON. Saving to", p)
        with open(p, "w") as f:
            f.write(text)
        # raise e
        return
    for item in outputs:
        c = batch[item["id"]]["meta"]
        with open(output_path(c), "w") as f:
            f.write(item["content"])
        print(f"  ✓ {output_path(c)} {len(item['content']):,} chars")
    if len(outputs) != len(batch):
        print(f"Warning: expected {len(batch)} outputs but got {len(outputs)}")
    print(f"Sample: {outputs[0]['content'][:200]}...")


with open("chapters_sample.json", "r") as f:
    chapters = json.load(f)

tasks = []
for c in chapters:
    if random.random() < 0.9:
        continue
    p = summary_path(c)
    if not os.path.exists(p) or os.path.exists(output_path(c)):
        continue
    with open(p, "r") as f:
        summary = f.read().strip()
    if not summary:
        continue
    tasks.append({"meta": c, "summary": summary})

batches = [tasks[i : i + BATCH] for i in range(0, len(tasks), BATCH)]
print(f"Total: {len(tasks)} summaries in {len(batches)} batches")

with ThreadPoolExecutor(max_workers=args.worker) as executor:
    stats = {"recent": deque()}
    lock = Lock()
    futures = [executor.submit(generate_batch, b) for b in batches]
    for f in (pbar := tqdm(as_completed(futures), total=len(futures))):
        try:
            f.result()
        except Exception as e:
            print(f"Error: {e}")
            # raise e
            if '429' in str(e):
                import os
                os._exit(1)
            time.sleep(60)
