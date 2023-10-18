# Combined
import glob
import pandas as pd
from collections import Counter
from functools import reduce

def merge(x):
    results = reduce(lambda x,y:x+y, map(lambda x: list(filter(None, x.lower().split('|'))),x[1:].tolist()))
    c = Counter(results)
    out = {
        f'merge_{thu}':
        '|'.join(set(filter(lambda x: c[x] >= thu, c)))
        for thu in range(1,7)
    }
    return pd.Series(out)


for dsname in ['HTGen6k', 'HTGen12k', 'HTName', 'HTGenV2']:
    combined_csv = pd.concat([pd.read_csv(f).rename(columns={'pred':f.split('_')[-1].replace('.csv','')+'_pred'}) for f in glob.glob(f'{dsname}*.csv') if "result" not in f], axis=1) 

    combined_csv = combined_csv.T.drop_duplicates(keep='first').T
    # combined_csv = combined_csv.drop(columns=['Unnamed: 0'])
    res = pd.concat([combined_csv, combined_csv.fillna('').apply(merge, axis=1)], axis=1)
    dsname = dsname.replace('_','')
    res.to_csv(f'{dsname}_finetune_result.csv', index=False) # .columns #

    c = list(combined_csv.columns)[1:]
    c.sort()
    fdict = {
        'HTName': './data/HT/HTName.csv',
        'HTGen6k': './HTGen6k_deberta-v3-base-conll2003.csv',
        'HTGen12k': './HTGen12k_deberta-v3-base-conll2003.csv',
        'HTGenV2': './results/finetune/HTGenV2_deberta-v3-base-conll2003.csv'
    }
    print("Run the following command to get the scores:")
    print(f"echo -e '{dsname}' >> results/scores.txt && python3 src/neat_metrics.py --prediction './results/finetune/{dsname}_finetune_result.csv' --ground_truth '{fdict[dsname]}' --prediction_column {' '.join(c)} --ground_truth_column {' '.join(['gpt_name']*len(c))} >> results/scores.txt")