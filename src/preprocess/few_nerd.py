import pandas as pd
import os
from os.path import join as pj

data_path = "./data/Few-NERD/"
cache_path = "./data/cache/Few-NERD/"
os.makedirs(cache_path, exist_ok=True)
files = ["train.txt", "test.txt", "dev.txt"]

for file in files:
    with open(pj(data_path, file), "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    texts, tags, tags_l1 = [[]], [[]], [[]]

    for line in lines:
        if line:
            text, tag = line.split("\t")
            texts[-1].append(text)
            tags[-1].append(tag)
            tags_l1[-1].append(tag.split("-")[0])
        else:
            texts.append([])
            tags.append([])
            tags_l1.append([])

    pd.DataFrame({"tokens": texts, "tags": tags}).to_csv(
        pj(cache_path, file.replace(".txt", ".csv")), index=False
    )
    pd.DataFrame({"tokens": texts, "tags": tags_l1}).to_csv(
        pj(cache_path, file.replace(".txt", "_L1.csv")), index=False
    )
