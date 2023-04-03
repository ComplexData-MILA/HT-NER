# tdf = pd.DataFrame({"a": [1, 2, 3], "b": ['hello', 'ola', 'thammi']})
# vdf = pd.DataFrame({"a": [4, 5, 6], "b": ['four', 'five', 'six']})
# tds = Dataset.from_pandas(tdf)
# vds = Dataset.from_pandas(vdf)


# ds = DatasetDict()

# ds['train'] = tds
# ds['validation'] = vds

# print(ds)# tdf = pd.DataFrame({"a": [1, 2, 3], "b": ['hello', 'ola', 'thammi']})
# vdf = pd.DataFrame({"a": [4, 5, 6], "b": ['four', 'five', 'six']})
# tds = Dataset.from_pandas(tdf)
# vds = Dataset.from_pandas(vdf)


# ds = DatasetDict()

# ds['train'] = tds
# ds['validation'] = vds

# print(ds)

import pandas as pd

files = ['./train.txt', './test.txt', './dev.txt']

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    
    texts, tags, tags_l1 = [[]], [[]], [[]]

    for line in lines:
        if line:
            text, tag = line.split('\t')
            texts[-1].append(text)
            tags[-1].append(tag)
            tags_l1[-1].append(tag.split('-')[0])
        else:
            texts.append([])
            tags.append([])
            tags_l1.append([])
            
    pd.DataFrame({'text': texts, 'tag': tags}).to_csv(file.replace('.txt', '.csv'), index=False)
    pd.DataFrame({'text': texts, 'tag': tags_l1}).to_csv(file.replace('.txt', '_l1.csv'), index=False)
