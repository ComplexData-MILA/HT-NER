import bz2
import pandas as pd
import os
from os.path import join as pj

data_path = "data/wikiner-en/"
cache_path = "data/cache/wikiner-en/"
os.makedirs(cache_path, exist_ok=True)

with bz2.open(pj(data_path, "aij-wikiner-en-wp2.bz2"), "rb") as f:
    content = f.read().decode("utf-8").split("\n")

texts, tags = [], []
for sent in content:
    if not sent:
        continue
    text, tag = [], []
    for word in sent.split(" "):
        word, _, label = word.split("|")
        text, tag = text + [word], tag + [label.split("-")[-1]]
        # print(word, label)
    texts.append(text), tags.append(tag)

pd.DataFrame({"tokens": texts, "tags": tags}).to_csv(
    pj(cache_path, "aij-wikiner-en-wp2.csv"), index=False
)

# print(len(texts))
# print(set([y for x in tags for y in x]))
