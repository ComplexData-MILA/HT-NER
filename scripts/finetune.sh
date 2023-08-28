SEED=577069
# python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets wnut2017 --seed $SEED
python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets conll2003 --seed ${SEED}
# python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets fewnerd-l1 --seed $SEED
python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets wikiner-en --seed ${SEED}
# python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets HTUnsup
# python src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets HTGen

# python3 src/finetune.py --base-model "roberta-base" --datasets wnut2017 --seed $SEED
# # python3 src/finetune.py --base-model "roberta-base" --datasets fewnerd-l1
# python3 src/finetune.py --base-model "roberta-base" --datasets conll2003 --seed $SEED
python3 src/finetune.py --base-model "roberta-base" --datasets wikiner-en --seed ${SEED}
# python3 src/finetune.py --base-model "roberta-base" --datasets HTUnsup
# python3 src/finetune.py --base-model "roberta-base" --datasets HTGen

# Distributed training for polyglot_ner
# python3 -m torch.distributed.launch --nproc_per_node=2 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets polyglot_ner
# python3 -m torch.distributed.launch --nproc_per_node=2 src/finetune.py --base-model "roberta-base" --datasets polyglot_ner

# python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets ontonotes5
# python3 src/finetune.py --base-model "roberta-base" --datasets ontonotes5
