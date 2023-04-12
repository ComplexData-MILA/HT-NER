Calendar    https://docs.google.com/spreadsheets/d/1DLX0il3okq-H7Snnuz8liA4wFwxoqZRWYzVesoX8efk

OVERLEAF	https://www.overleaf.com/project/63ce0f2bdd6d9dd2b1dbc973

GITHUB Repo	https://github.com/ComplexData-MILA/HT-NER

**Dev Repo**

# The Name Extraction Framework for Combating Human Trafficking
## Author: 
- Javin: GPT-based name expansion and GPT based data-label generations
- Vidya: Aggregation
- Peter: ChatGPT/GPT4 based pseudo labels to finetune BERT
    - ChatGPT/GPT gerenating pseudo labels:
        PreCondition: ```export OPENAI_API_KEY='yourkey'```
        Usage: 
        ```
        openai_infer.py [-h] [--data DATA] [--save_path SAVE_PATH]
                      [--model {gpt4,gpt3.5,davinci,D,Curie,C,Babbage,B,Ada,A}] [--prompt PROMPT]
                      [--result_column RESULT_COLUMN] [--verbose]

        optional arguments:
            -h, --help            show this help message and exit
            --data DATA           dataset-name | csv_file[have 'text' coloum]
            --save_path SAVE_PATH
            --model {gpt4,gpt3.5,davinci,D,Curie,C,Babbage,B,Ada,A}
            --prompt PROMPT       default: see code
            --result_column_name RESULT_COLUMN_NAME
            --verbose
        ```

    - Fientune DeBERTav3:
        Usage:
        ```
        python3 src/finetune.py 
            --data dataset-name | csv_file[have 'text' coloum]
            --label_column gpt_name gpt_location gpt_social_media
            --save_dir ./models
        ```

    - Evaluate with Finetuned Model:
        Usage:
        ```
        python3 src/inference.py 
            --data dataset-name | csv_file[have 'text' coloum]
            --label_column gpt_name gpt_location gpt_social_media
            --save_path ./prediction/HT1K_finetune.csv
        ```

    - Evaluate F1:
        ```
        usage: metrics.py [-h] [--ground_truth GROUND_TRUTH]
                        [--ground_truth_column GROUND_TRUTH_COLUMN [GROUND_TRUTH_COLUMN ...]]
                        [--pred PRED]
                        [--prediction_column PREDICTION_COLUMN [PREDICTION_COLUMN ...]]

        Evaluate F1 score for each column in entity and token
        levels.

        optional arguments:
        -h, --help            show this help message and exit
        --ground_truth GROUND_TRUTH
                                Name and location of ground truth
                                CSV file.
        --ground_truth_column GROUND_TRUTH_COLUMN [GROUND_TRUTH_COLUMN ...]
                                Names of columns in ground truth
                                CSV file that contain the
                                entities.
        --pred PRED           Location of prediction CSV file.
        --prediction_column PREDICTION_COLUMN [PREDICTION_COLUMN ...]
                                Names of columns in prediction CSV
                                file that contain the entities.
        ```
        <!-- python3 src/evalute.py \
            --ground_truth dataset-name | csv_file
            --ground_truth_column gpt_name gpt_location gpt_social_media
            --pred './prediction/HT1K_finetune.csv'
            --predition_column name location social_media -->

    - Full Command:
        ```
        # ChatGPT on HTUnsup

        python3 src/openai_infer.py \
            --data ./data/HTUnsup.csv \
            --save_path ./results/HTUnsup_chatgpt.csv \
            --result_column chatgpt_response \
            --model gpt3.5
        
        # ChatGPT on HTName

        python3 src/openai_infer.py \
            --data ./data/HTName.csv \
            --save_path ./data/HTName_chatgpt.csv \
            --result_column chatgpt_response \
            --model gpt3.5

        python3 src/metrics.py \
            --ground_truth ./results/HTName_chatgpt.csv  \
            --ground_truth_column label \
            --prediction ./results/HTName_chatgpt.csv \
            --prediction_column gpt_name

        # ChatGPT on HTUnified

        python3 src/openai_infer.py \
            --data ./data/HTUnified.csv \
            --save_path ./results/HTUnified_chatgpt.csv \
            --result_column chatgpt_response \
            --model gpt3.5

        python3 src/metrics.py \
            --ground_truth ./results/HTUnified_chatgpt.csv  \
            --ground_truth_column name \
            --prediction ./results/HTUnified_chatgpt.csv \
            --prediction_column gpt_name

        python3 src/metrics.py \
            --ground_truth ./results/HTUnified_chatgpt.csv  \
            --ground_truth_column location \
            --prediction ./results/HTUnified_chatgpt.csv \
            --prediction_column gpt_location

        # Data Preprocess

        python3 src/preprocess/human_trafficking.py
        python3 src/preprocess/few_nerd.py
        python3 src/preprocess/wikiner.py

        # Verify Dataset
        python3 src/dataset.py

        # Finetune DeBERTav3
        CUDA_VISIBLE_DEVICES=0 python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets wnut2017
        python3 src/finetune_evaluation.py --base-model /home/mila/h/hao.yu/ht/HT-NER/saved_models/deberta-v3-base-wnut2017/checkpoint-170 --dataset wnut2017

        CUDA_VISIBLE_DEVICES=0 python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets HTUnsup
        python3 src/finetune_evaluation.py --base-model /home/mila/h/hao.yu/ht/HT-NER/saved_models/deberta-v3-base-HTUnsup/checkpoint-1540 --dataset HTUnsup # default use cuda:0
        
        ```

File Structure After Data Preprocess:
```
    data
    ├── cache
    │   ├── Few-NERD
    │   │   ├── dev.csv
    │   │   ├── dev_L1.csv
    │   │   ├── test.csv
    │   │   ├── test_L1.csv
    │   │   ├── train.csv
    │   │   └── train_L1.csv
    │   ├── HT
    │   │   ├── HTName_tokenized.csv
    │   │   ├── HTUnified_tokenized.csv
    │   │   └── HTUnsup_tokenized.csv
    │   └── wikiner-en
    │       └── aij-wikiner-en-wp2.csv
    ├── Few-NERD
    │   ├── dev.txt
    │   ├── supervised.zip
    │   ├── test.txt
    │   └── train.txt
    ├── HT
    │   ├── HTName.csv
    │   ├── HTUnified.csv
    │   └── HTUnsup.csv
    └── wikiner-en
        └── aij-wikiner-en-wp2.bz2

```

## Paper
