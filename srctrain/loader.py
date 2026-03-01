import json
import os
from sklearn.model_selection import GroupShuffleSplit
import re

models = [
    'gemini',  # Gemini 3 Pro
    'qwen',  # Qwen coder plus
    'pony',  # GLM-5
    'kimi25',  # Kimi 2.5
    'glm47',  # GLM-4.7
    'doubao',  # Doubao code
    'deepseekv32',  # DeepSeek v3.2
]

gen_folders = {k: i for i, k in enumerate(models, start=1)}

def load_dataset():
    print("loading real chapters...")
    with open('chapters_sample.json', 'r') as f:
        chapters = json.load(f)
    print("loading generated chapters...")
    samples = []
    for c in chapters:
        if 'content' not in c:
            continue
        content = c['content'].replace('&lt;br&gt;', '\n').replace('\u3000', ' ')
        if len(content.strip()) <= 1000:
            continue
        
        sample = {
            'human': content.strip(),
            'novel_id': c['novelId'],
            'chapter': c['chapterId']
        }
        for k in gen_folders.keys():
            output_path = f"generated_{k}/{c['novelId']}_{c['chapterId']}.txt"
            if not os.path.exists(output_path):
                continue
        
            with open(output_path, 'r', encoding='utf-8') as f:
                t = f.read().strip()
            
            sample[k] = t
        
        # all models
        # if len(sample) != 3 + len(gen_folders):
        #     continue

        # ANY model
        if len(sample) <= 3:  # 只有 human / novel_id / chapter，无任何模型数据
            continue

        samples.append(sample)
    print(f"loaded {len(samples)} samples")
    
    splitter = GroupShuffleSplit(test_size=.2, n_splits=1, random_state=42)
    split = splitter.split(samples, groups=list(map(lambda x: x['novel_id'], samples)))
    train_idx, test_idx = next(split)
    print(f"train chapter size: {len(train_idx)}, test chapter size: {len(test_idx)}")
    return [samples[i] for i in train_idx], [samples[i] for i in test_idx]



def split_chinese_sentence(s, min_chars=5):
    pattern = re.compile(r'([。！？；?!\n])')
    cn_ptrn = re.compile(r'[^a-zA-Z0-9\n,.;?!\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef]')
    s = re.sub(cn_ptrn, '', s)
    sentences = pattern.split(s)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= min_chars]
    return sentences


def to_col(samples, only_model=None):
    x = []
    y = []
    keys = [only_model] if only_model is not None else gen_folders.keys()
    for i in samples:
        for s in split_chinese_sentence(i['human']):
            x.append(s)
            # print(s)
            y.append(0)
        for k in keys:
            if k in i:
                for s in split_chinese_sentence(i[k]):
                    x.append(s)
                    # print(s)
                    y.append(gen_folders[k])
    return x, y

if __name__ == "__main__":
    load_dataset()