# SEED=577069
# python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets wnut2017 --seed ${SEED}
# python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets conll2003 --seed ${SEED}
# python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets wikiner-en --seed ${SEED}

# python3 src/finetune.py --base-model "roberta-base" --datasets wnut2017 --seed ${SEED}
# python3 src/finetune.py --base-model "roberta-base" --datasets conll2003 --seed ${SEED}
# python3 src/finetune.py --base-model "roberta-base" --datasets wikiner-en --seed ${SEED}

export SEED=5770691
python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets wnut2017 --seed ${SEED}
python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets conll2003 --seed ${SEED}
python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets wikiner-en --seed ${SEED}

python3 src/finetune.py --base-model "roberta-base" --datasets wnut2017 --seed ${SEED}
python3 src/finetune.py --base-model "roberta-base" --datasets conll2003 --seed ${SEED}
python3 src/finetune.py --base-model "roberta-base" --datasets wikiner-en --seed ${SEED}

export SEED=5770692
python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets wnut2017 --seed ${SEED}
python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets conll2003 --seed ${SEED}
python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets wikiner-en --seed ${SEED}

python3 src/finetune.py --base-model "roberta-base" --datasets wnut2017 --seed ${SEED}
python3 src/finetune.py --base-model "roberta-base" --datasets conll2003 --seed ${SEED}
python3 src/finetune.py --base-model "roberta-base" --datasets wikiner-en --seed ${SEED}

export SEED=5770693
python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets wnut2017 --seed ${SEED}
python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets conll2003 --seed ${SEED}
python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets wikiner-en --seed ${SEED}

python3 src/finetune.py --base-model "roberta-base" --datasets wnut2017 --seed ${SEED}
python3 src/finetune.py --base-model "roberta-base" --datasets conll2003 --seed ${SEED}
python3 src/finetune.py --base-model "roberta-base" --datasets wikiner-en --seed ${SEED}

export SEED=5770694
python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets wnut2017 --seed ${SEED}
python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets conll2003 --seed ${SEED}
python3 src/finetune.py --base-model "microsoft/deberta-v3-base" --datasets wikiner-en --seed ${SEED}

python3 src/finetune.py --base-model "roberta-base" --datasets wnut2017 --seed ${SEED}
python3 src/finetune.py --base-model "roberta-base" --datasets conll2003 --seed ${SEED}
python3 src/finetune.py --base-model "roberta-base" --datasets wikiner-en --seed ${SEED}


# python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-wnut2017/checkpoint-170 --datasets wnut2017
# python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-conll2003/checkpoint-705 --datasets conll2003
# python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-wikiner-en/checkpoint-4515 --datasets wikiner-en