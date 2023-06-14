Calendar    https://docs.google.com/spreadsheets/d/1DLX0il3okq-H7Snnuz8liA4wFwxoqZRWYzVesoX8efk

OVERLEAF	https://www.overleaf.com/project/63ce0f2bdd6d9dd2b1dbc973

GITHUB Repo	https://github.com/ComplexData-MILA/HT-NER

**Dev Repo**

# The Name Extraction Framework for Combating Human Trafficking
## Author: 
- Javin: GPT-based name expansion and GPT based data-label generations
- Vidya: Aggregation
- Peter: ChatGPT/GPT4 based pseudo labels to finetune BERT
    - <details>
        <summary>ChatGPT/GPT gerenating pseudo labels</summary>
        
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
        </detail>

    - <details>
        <summary>Fientune DeBERTav3</summary>

        Usage:
        ```
        python3 src/finetune.py 
            --data dataset-name | csv_file[have 'text' coloum]
            --label_column gpt_name gpt_location gpt_social_media
            --save_dir ./models
        ```
        </detail>

    - <details>
        <summary>Evaluate with Finetuned Model:</summary>

        Usage:
        ```
        python3 src/inference.py 
            --data dataset-name | csv_file[have 'text' coloum]
            --label_column gpt_name gpt_location gpt_social_media
            --save_path ./prediction/HT1K_finetune.csv
        ```
        </detail>

    - <details>
        <summary>Evaluate F1 Score</summary>

        Usage:
        ```
        python3 neat_metrics.py [-h] [--ground_truth GROUND_TRUTH]
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
        </detail>

    - <details>
        <summary>ChatGPT Full Command</summary>

        ```bash
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

        ```
        </details>
        
        <details>
        <summary>Finetune Full Command</summary>
        
        Reference: https://github.com/huggingface/peft#token-classification
        
        ### BaseCommand
        ```bash
        # Finetune:
        CUDA_VISIBLE_DEVICES=0 python3 src/finetune.py --base-model [model_name] --datasets wnut2017
        CUDA_VISIBLE_DEVICES=0 python3 src/finetune.py --base-model [model_name] --datasets HTUnsup

        # Evaluation:
        python3 src/finetune_evaluation.py \
            --base-model ./saved_models/deberta-v3-base-wnut2017/checkpoint-170 \
            --dataset wnut2017
            
        python3 src/finetune_evaluation.py \
            --base-model ./saved_models/deberta-v3-base-HTUnsup/checkpoint-1540\
            --dataset HTUnsup
        ```

        ### BERT Series
        - DeBERTav3
            ```model_name = "microsoft/deberta-v3-base"```
        - RoBERTa
            ```model_name = "roberta-base"```
        - BERT
            ```model_name = "bert-base-uncased"```

        ### GPT Series
        - GPT2
            ```model_name = "gpt2"```
        - BLOOM-560M
            ```model_name = "bigscience/bloom-560m"```
        - OPT-350M
            ```model_name = "facebook/opt-350m"```

        </details>

    More and detailed exctuable commands for training can be found in [finetune.sh](./scripts/finetune.sh), for evaluting can be found in [eval.sh](./scripts/eval.sh)
