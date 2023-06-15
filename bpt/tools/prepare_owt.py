from datasets import load_dataset
import json
from multiprocessing import Pool, cpu_count

dataset = load_dataset("openwebtext")

split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

def save_split(split):
    with open(f"openwebtext_{split}.jsonl", "w") as f:
        for example in split_dataset[split]:
            json.dump({"text": example["text"]}, f)
            f.write("\n")

with Pool(cpu_count()) as p:
    p.map(save_split, ["train", "val"])
