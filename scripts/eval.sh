# Eval on CoNLL-2003, WNUT-2017, FewNERD-L1, WikiNER-EN
python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-wnut2017/checkpoint-170 --datasets wnut2017
python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-conll2003/checkpoint-705 --datasets conll2003
python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-fewnerd-l1/checkpoint-32945 --datasets fewnerd-l1
python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-wikiner-en/checkpoint-4515 --datasets wikiner-en
python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-HTUnsup/checkpoint-1540 --datasets HTUnsup
python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-polyglot_ner/checkpoint-7505 --datasets polyglot_ner
python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-HTGen/checkpoint-730 --datasets HTGen
python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-HTGen12k/checkpoint-1340 --datasets HTGen12k
python3 src/finetune_evaluation.py --base-model ./saved_models/deberta-v3-base-HTGenV2/checkpoint-1565 --datasets HTGenV2

python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-wnut2017/checkpoint-170 --datasets wnut2017
python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-conll2003/checkpoint-705 --datasets conll2003
python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-wikiner-en/checkpoint-4515 --datasets wikiner-en
python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-HTUnsup/checkpoint-1540 --datasets HTUnsup
python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-fewnerd-l1/checkpoint-32945 --datasets fewnerd-l1
python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-polyglot_ner/checkpoint-3755 --datasets polyglot_ner
python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-HTGen/checkpoint-730 --datasets HTGen
python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-HTGen12k/checkpoint-1340 --datasets HTGen12k
python3 src/finetune_evaluation.py --base-model ./saved_models/roberta-base-HTGenV2/checkpoint-1565 --datasets HTGenV2


# Eval on HTName/HTUnified/HTGen
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-wnut2017/checkpoint-170 --datasets wnut2017
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-conll2003/checkpoint-705 --datasets conll2003
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-fewnerd-l1/checkpoint-32945 --datasets fewnerd-l1
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-wikiner-en/checkpoint-4515 --datasets wikiner-en
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-HTUnsup/checkpoint-1540 --datasets HTUnsup
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-polyglot_ner/checkpoint-7505 --datasets polyglot_ner
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-HTGen/checkpoint-730 --datasets HTGen
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-HTGen12k/checkpoint-1340 --datasets HTGen12k
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/deberta-v3-base-HTGenV2/checkpoint-1565 --datasets HTGenV2

python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-wnut2017/checkpoint-170 --datasets wnut2017
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-conll2003/checkpoint-705 --datasets conll2003
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-wikiner-en/checkpoint-4515 --datasets wikiner-en
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-HTUnsup/checkpoint-1540 --datasets HTUnsup
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-fewnerd-l1/checkpoint-32945 --datasets fewnerd-l1
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-polyglot_ner/checkpoint-3755 --datasets polyglot_ner
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-HTGen/checkpoint-730 --datasets HTGen
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-HTGen12k/checkpoint-1340 --datasets HTGen12k
python3 src/finetune_evaluation_HT.py --base-model ./saved_models/roberta-base-HTGenV2/checkpoint-1565 --datasets HTGenV2
