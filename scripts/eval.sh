python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-wnut2017/checkpoint-170 --datasets wnut2017
python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-conll2003/checkpoint-705 --datasets conll2003
python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-fewnerd-l1/checkpoint-32945 --datasets fewnerd-l1
python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-wikiner-en/checkpoint-4515 --datasets wikiner-en
python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-HTUnsup/checkpoint-1540 --datasets HTUnsup

python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-wnut2017/checkpoint-170 --datasets wnut2017
python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-conll2003/checkpoint-705 --datasets conll2003
python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-wikiner-en/checkpoint-4515 --datasets wikiner-en
python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-HTUnsup/checkpoint-1540 --datasets HTUnsup
python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-fewnerd-l1/checkpoint-32945 --datasets fewnerd-l1

python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-polyglot_ner/checkpoint-7505 --datasets polyglot_ner
python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-polyglot_ner/checkpoint-3755 --datasets polyglot_ner


python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-HTGen/checkpoint-200 --datasets HTGen
python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-HTGen/checkpoint-200 --datasets HTGen





python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-wnut2017/checkpoint-170 --datasets wnut2017
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-conll2003/checkpoint-705 --datasets conll2003
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-fewnerd-l1/checkpoint-32945 --datasets fewnerd-l1
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-wikiner-en/checkpoint-4515 --datasets wikiner-en
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-HTUnsup/checkpoint-1540 --datasets HTUnsup

python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-wnut2017/checkpoint-170 --datasets wnut2017
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-conll2003/checkpoint-705 --datasets conll2003
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-wikiner-en/checkpoint-4515 --datasets wikiner-en
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-HTUnsup/checkpoint-1540 --datasets HTUnsup
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-fewnerd-l1/checkpoint-32945 --datasets fewnerd-l1

python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-polyglot_ner/checkpoint-7505 --datasets polyglot_ner
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-polyglot_ner/checkpoint-3755 --datasets polyglot_ner



# python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-ontonotes5/checkpoint-955 --datasets ontonotes5

# updates
CUDA_VISIBLE_DEVICES=1 python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-wikiner-en/checkpoint-2310 --datasets wikiner-en
CUDA_VISIBLE_DEVICES=1 python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-wikiner-en/checkpoint-2310 --datasets wikiner-en

CUDA_VISIBLE_DEVICES=0 python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-wikiner-en/checkpoint-2310 --datasets wikiner-en
CUDA_VISIBLE_DEVICES=0 python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-wikiner-en/checkpoint-2310 --datasets wikiner-en