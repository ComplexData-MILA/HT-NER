# SWEET - Weakly Supervised Person Name Extraction for Fighting Human Trafficking

## Repository Contributor: 
Hao Yu, Javin Liu, Vidua Sujaya

- Hao Yu: ChatGPT/GPT4 based pseudo labels to finetune BERT
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

    More exctuable commands for training can be found in [finetune.sh](./scripts/finetune.sh), for evaluting can be found in [eval.sh](./scripts/eval.sh)

- Javin Liu: GPT-based name expansion and GPT based data-label generations
- Vidya: Aggregation

|                | Evluationg Dataset |      | Broad Twitter Corpus |        |      | Tweebank-NER |        |
|----------------|--------------------|:----:|:--------------------:|:------:|:----:|:------------:|:------:|
| Model          | Training Dataset   | F1   | Precision            | Recall | F1   | Precision    | Recall |
| DeBERTaV3-base | HTUnsup            | 0.50 |                 0.81 |   0.36 | 0.41 |         0.40 |   0.41 |
| DeBERTaV3-base | CoNLL2003          | 0.49 |                 0.77 |   0.35 | 0.63 |         0.51 |   0.82 |
| DeBERTaV3-base | Few-NERD-L1        | 0.38 |                 0.85 |   0.24 | 0.79 |         0.88 |   0.72 |
| DeBERTaV3-base | WikiNER-en         | 0.53 |                 0.79 |   0.40 | 0.59 |         0.49 |   0.76 |
| DeBERTaV3-base | WNUT2017           | 0.37 |                 0.76 |   0.24 | 0.75 |         0.69 |   0.82 |
| RoBERTa-base   | HTUnsup            | 0.67 |                 0.84 |   0.56 | 0.42 |         0.33 |   0.57 |
| RoBERTa-base   | CoNLL2003          | 0.51 |                 0.70 |   0.41 | 0.58 |         0.44 |   0.82 |
| RoBERTa-base   | Few-NERD-L1        | 0.46 |                 0.76 |   0.33 | 0.78 |         0.87 |   0.71 |
| RoBERTa-base   | WikiNER-en         | 0.53 |                 0.83 |   0.39 | 0.51 |         0.39 |   0.75 |
| RoBERTa-base   | WNUT2017           | 0.40 |                 0.87 |   0.26 | 0.81 |         0.81 |   0.80 |