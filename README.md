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
        openai_gpt.py [-h] [--data DATA] [--save_path SAVE_PATH]
                      [--model {gpt4,gpt3.5,davinci,D,Curie,C,Babbage,B,Ada,A}] [--prompt PROMPT]
                      [--result_column RESULT_COLUMN] [--verbose]
        ```
        ```
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

    - Inference with Finetuned Model:
        Usage:
        ```
        python3 src/inference.py 
            --data dataset-name | csv_file[have 'text' coloum]
            --label_column gpt_name gpt_location gpt_social_media
            --save_path ./prediction/HT1K_finetune.csv
        ```

    - Evaluate F1:
        ```
        usage: evaluate.py [-h] [--ground_truth GROUND_TRUTH]
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

        python3 src/openai_gpt.py \
            --data ./data/HTUnsup.csv \
            --save_path ./results/HTUnsup_chatgpt.csv \
            --result_column chatgpt_response \
            --model gpt3.5
        
        python3 src/process_response.py \
            --data ./results/HTUnsup_chatgpt.csv \
            --save_path ./results/HTUnsup_chatgpt.csv


        # ChatGPT on HTName

        python3 src/openai_gpt.py \
            --data ./data/HTName.csv \
            --save_path ./data/HTName_chatgpt.csv \
            --result_column chatgpt_response \
            --model gpt3.5

        python3 src/process_response.py \
            --data ./results/HTName_chatgpt.csv \
            --save_path ./results/HTName_chatgpt.csv

        python3 src/evaluate.py \
            --ground_truth ./results/HTName_chatgpt.csv  \
            --ground_truth_column label \
            --prediction ./results/HTName_chatgpt.csv \
            --prediction_column gpt_name

        # ChatGPT on HTLocation

        python3 src/openai_gpt.py \
            --data ./data/HTLocation.csv \
            --save_path ./results/HTLocation_chatgpt.csv \
            --result_column chatgpt_response \
            --model gpt3.5

        python3 src/process_response.py \
            --data ./results/HTLocation_chatgpt.csv \
            --save_path ./results/HTLocation_chatgpt.csv

        python3 src/evaluate.py \
            --ground_truth ./results/HTLocation_chatgpt.csv  \
            --ground_truth_column label \
            --prediction ./results/HTLocation_chatgpt.csv \
            --prediction_column gpt_name

        ```

File Structure:
```
    ./data
    ├── Few-NERD
    │   ├── dev.txt
    │   ├── test.txt
    │   └── train.txt
    ├── ht1k_self_labeled
    │   ├── seed0
    │   │   ├── HT1K_test.tsv
    │   │   └── HT1K_test.xlsx
    │   ├── seed1
    │   │   └── HT1K_test.tsv
    │   ├── seed2
    │   │   └── HT1K_test.tsv
    │   ├── seed3
    │   │   └── HT1K_test.tsv
    │   └── seed4
    │       └── HT1K_test.tsv
    ├── ht_tokenized.csv
    ├── ht_tokenized_street.csv
    ├── location_eval_500ads.csv
    ├── oda
    │   ├── AB_processed_streets.json
    │   ├── BC_processed_streets.json
    │   ├── MB_processed_streets.json
    │   ├── NB_processed_streets.json
    │   ├── NS_processed_streets.json
    │   ├── NT_processed_streets.json
    │   ├── ON_processed_streets.json
    │   ├── PE_processed_streets.json
    │   ├── QC_processed_streets.json
    │   └── SK_processed_streets.json
    ├── revision_experiments_data_with_5_splits
    │   ├── seed0
    │   │   ├── conll_test_baseline.tsv
    │   │   ├── conll_test.tsv
    │   │   ├── conll_test.txt
    │   │   ├── conll_train.tsv
    │   │   ├── conll_train.txt
    │   │   ├── HT1k_test_baseline.tsv
    │   │   ├── HT1k_test_baseline.xlsx
    │   │   ├── HT1K_test_new.tsv
    │   │   ├── HT1K_test.tsv
    │   │   ├── HT1K_test.xlsx
    │   │   ├── HT1K_train.tsv
    │   │   ├── HT2k_test_baseline.tsv
    │   │   ├── HT2k_test.tsv
    │   │   ├── HT2k_train.tsv
    │   │   ├── listcrawler_test.tsv
    │   │   ├── listcrawler_test.txt
    │   │   ├── listcrawler_train.tsv
    │   │   ├── listcrawler_train.txt
    │   │   ├── locanto_test.tsv
    │   │   ├── locanto_test.txt
    │   │   ├── locanto_train.tsv
    │   │   ├── locanto_train.txt
    │   │   ├── wnut_test_baseline.tsv
    │   │   ├── wnut_test.tsv
    │   │   ├── wnut_test.txt
    │   │   ├── wnut_train.tsv
    │   │   └── wnut_train.txt
    │   ├── seed0_HT1K_test_new.tsv
    │   ├── seed1 2 3 4 
    │       ....
    └── wikiner-en
        ├── aij-wikiner-en-wp2.bz2
        ├── aij-wikiner-en-wp2.csv
        └── aij-wikiner-en-wp3.bz2
```

## Paper
