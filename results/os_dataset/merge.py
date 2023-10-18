import glob
import pandas as pd

for dsname in ['conll2003', 'wnut2017', 'fewnerdl1', 'wikiner','btwitter', 'tweebank']:
    combined_csv = pd.concat([pd.read_csv(f).rename(columns={'pred':f.split('_')[1].replace('.csv','')+'_pred'}) for f in glob.glob(f'{dsname}*.csv') if 'result' not in f], axis=1)

    combined_csv = combined_csv.T.drop_duplicates(keep='first').T
    combined_csv = combined_csv.drop(columns=['Unnamed: 0'])
    combined_csv.to_csv(f'{dsname}_fintune_result.csv', index=False)
    c = list(combined_csv.columns)[1:]
    c.sort()
    print(f"echo -e '{dsname}' >> results/scores.txt && python3 src/neat_metrics.py --prediction {f'./results/finetune2/{dsname}_fintune_result.csv'} --ground_truth {f'./results/finetune2/{dsname}_fintune_result.csv'} --prediction_column {' '.join(c)} --ground_truth_column {' '.join(['label']*len(c))} >> results/scores.txt")
