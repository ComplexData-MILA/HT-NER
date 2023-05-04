CUDA_VISIBLE_DEVICES=0 python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets wnut2017
CUDA_VISIBLE_DEVICES=0 python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets conll2003
CUDA_VISIBLE_DEVICES=0 python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets fewnerd-l1
CUDA_VISIBLE_DEVICES=0 python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets wikiner-en
CUDA_VISIBLE_DEVICES=0 python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets HTUnsup
CUDA_VISIBLE_DEVICES=0 python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets polyglot_ner


python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-wnut2017/checkpoint-170 --datasets wnut2017
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-conll2003/checkpoint-705 --datasets conll2003
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-fewnerd-l1/checkpoint-32945 --datasets fewnerd-l1
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-wikiner-en/checkpoint-6500 --datasets wikiner-en
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-HTUnsup/checkpoint-1540 --datasets HTUnsup

python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-wnut2017/checkpoint-170 --datasets wnut2017
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-conll2003/checkpoint-705 --datasets conll2003
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-wikiner-en/checkpoint-6500 --datasets wikiner-en
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-HTUnsup/checkpoint-1540 --datasets HTUnsup
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-fewnerd-l1/checkpoint-32945 --datasets fewnerd-l1


CUDA_VISIBLE_DEVICES=0 python3 src/finetune.py --base-model "roberta-base" --datasets wnut2017
CUDA_VISIBLE_DEVICES=0 python3 src/finetune.py --base-model "roberta-base" --datasets fewnerd-l1

CUDA_VISIBLE_DEVICES=1 python3 src/finetune.py --base-model "roberta-base" --datasets conll2003
CUDA_VISIBLE_DEVICES=1 python3 src/finetune.py --base-model "roberta-base" --datasets wikiner-en
CUDA_VISIBLE_DEVICES=0 python3 src/finetune.py --base-model "roberta-base" --datasets HTUnsup


python -m torch.distributed.launch --nproc_per_node=2 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets polyglot_ner
python -m torch.distributed.launch --nproc_per_node=2 src/finetune.py --base-model "roberta-base" --datasets polyglot_ner

python src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets ontonotes5
python src/finetune.py --base-model "roberta-base" --datasets ontonotes5

python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-polyglot_ner/checkpoint-16565 --datasets polyglot_ner
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-polyglot_ner/checkpoint-16565 --datasets polyglot_ner




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

