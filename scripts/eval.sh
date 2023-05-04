python3 src/finetune_infer.py --base-model ./saved_models/deberta-v3-base-wnut2017/checkpoint-170 --datasets wnut2017
python3 src/finetune_infer.py --base-model ./saved_models/deberta-v3-base-conll2003/checkpoint-705 --datasets conll2003
python3 src/finetune_infer.py --base-model ./saved_models/deberta-v3-base-fewnerd-l1/checkpoint-32945 --datasets fewnerd-l1
python3 src/finetune_infer.py --base-model ./saved_models/deberta-v3-base-wikiner-en/checkpoint-6500 --datasets wikiner-en
python3 src/finetune_infer.py --base-model ./saved_models/deberta-v3-base-HTUnsup/checkpoint-1540 --datasets HTUnsup

python3 src/finetune_infer.py --base-model ./saved_models/roberta-base-wnut2017/checkpoint-170 --datasets wnut2017
python3 src/finetune_infer.py --base-model ./saved_models/roberta-base-conll2003/checkpoint-705 --datasets conll2003
python3 src/finetune_infer.py --base-model ./saved_models/roberta-base-wikiner-en/checkpoint-6500 --datasets wikiner-en
python3 src/finetune_infer.py --base-model ./saved_models/roberta-base-HTUnsup/checkpoint-1540 --datasets HTUnsup
python3 src/finetune_infer.py --base-model ./saved_models/roberta-base-fewnerd-l1/checkpoint-32945 --datasets fewnerd-l1

