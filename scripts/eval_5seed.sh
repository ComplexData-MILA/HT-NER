# Eval on CoNLL-2003, WNUT-2017, FewNERD-L1, WikiNER-EN
python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-${SEED}-wnut2017/checkpoint-135 --datasets wnut2017
python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-${SEED}-conll2003/checkpoint-550 --datasets conll2003
python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-${SEED}-wikiner-en/checkpoint-9025 --datasets wikiner-en

python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-${SEED}-wnut2017/checkpoint-135 --datasets wnut2017
python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-${SEED}-conll2003/checkpoint-550 --datasets conll2003
python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-${SEED}-wikiner-en/checkpoint-9025 --datasets wikiner-en

