# https://github.com/hiyouga/LLaMA-Efficient-Tuning.git
import warnings
warnings.filterwarnings('ignore')
import os

cache_path = os.path.join(os.getenv("SCRATCH"), ".cache", "huggingface")
print("Cache path: ", cache_path)
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path

import wandb
import numpy as np
from dataset import loadDataset, ROOTS

from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    AutoModelForCausalLM
)


### Receive Augmentation
parser = argparse.ArgumentParser()
parser.add_argument("--base-model", type=str, required=True, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--datasets", nargs="+", type=str, required=True, default=['conll2003'])
parser.add_argument("--only-loc", type=int, default=0)
parser.add_argument("--fold", type=int, default=-1)
parser.add_argument("--sub-structure", type=str, default="")
parser.add_argument("--substitude", type=int, default=0)
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args(args=['--base-model','baichuan-inc/Baichuan-13B-Chat','--datasets','conll2003','--only-loc','0','--fold','-1','--substitude','0','--local_rank','-1'])

model_checkpoint = args.base_model
sub_structure = ""
sub_structure += "-LOC" if args.only_loc else ""
sub_structure += "" if args.fold == -1 else "-fold" + str(args.fold)
sub_structure += (
    ("-" + args.sub_structure)
    if args.sub_structure and args.sub_structure != "None"
    else ""
)

dataset_name = args.datasets[0]
if "HT" in dataset_name:
    batch_size = 32
elif "fewner" in dataset_name:
    batch_size = 32
elif "polyglot" in dataset_name:
    batch_size = 32
elif "ontonotes5" in dataset_name:
    batch_size = 32
else:
    batch_size = 128

lr = 2e-5 / 128 * batch_size

datasets, label_list, label_col_name = loadDataset(
    dataset_name,
    ROOTS(dataset_name),
    substitude=args.substitude,
    onlyLoc=args.only_loc,
)

l2id = {x: i for i, x in enumerate(label_list)}
id2l = dict(enumerate(label_list))
print(label_list)

wandb.init(
    project="HT-NER-Combined",
    name=f"{model_checkpoint.split('/')[-1]}{sub_structure}-{dataset_name}",
)
