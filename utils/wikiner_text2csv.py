import bz2
import pandas as pd

with bz2.open("data/wikiner-en/aij-wikiner-en-wp2.bz2", "rb") as f:
    content = f.read().decode('utf-8').split('\n')

texts, tags = [], []
# print(content[:20])
for sent in content:
    if not sent: continue
    text, tag = [], []
    for word in sent.split(" "):
        word, _, label = word.split("|")
        text, tag = text+[word], tag+[label.split('-')[-1]]
        # print(word, label)
    texts.append(text), tags.append(tag)
    
pd.DataFrame({'tokens': texts, 'tags': tags}).to_csv("data/wikiner-en/aij-wikiner-en-wp2.csv", index=False)


print(len(texts))
print(set([y for x in tags for y in x]))    
