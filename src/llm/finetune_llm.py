# %%
import os, argparse, wandb

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_SILENT"] = "true"
cache_path = os.path.join(os.getenv("SCRATCH"), ".cache", "huggingface")
print("Cache path: ", cache_path)
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
# with open(cache_path+"/token", "r") as f:
#     hf_token = f.read()
hf_token = True

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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig
import torch.nn as nn

#使用QLoRA引入的 NF4量化数据类型以节约显存
# model_name_or_path ='../baichuan-13b' #远程 'baichuan-inc/Baichuan-13B-Chat'
model_name_or_path = model_checkpoint
bnb_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

tokenizer = AutoTokenizer.from_pretrained(
   model_name_or_path, trust_remote_code=True, token=hf_token)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                quantization_config=bnb_config,
                token=hf_token,
                trust_remote_code=True) 

model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

prefix = '''Named entity recognition: Extract the three types of named entities in the text: name, location, and organization, and return the results in json format.

Here are some examples:

Xiaoming said to Xiaohong: "Have you heard of Amway?" -> {"person name": ["Xiaoming","Xiaohong"], "organization": ["Amway"]}
Now, hundreds of thousands of Chinese people visit the United States every year, and thousands of Chinese students study in the United States. -> {"Location": ["China", "USA"]}
China is one of the permanent members of the UN Security Council. -> {"Location": ["China"], "Organization": ["United Nations"]}

Please perform entity extraction on the following text and return it in json format.
'''
# %%
def get_prompt(text):
    return prefix+text+' -> '

def get_message(prompt,response):
    return [{"role": "user", "content": f'{prompt} -> '},
            {"role": "assistant", "content": response}]

messages = [{"role": "user", "content": get_prompt("Some Moroccan fans couldn't help but cheered in the stands")}]
response = model. chat(tokenizer, messages)
print(response)
messages = messages+[{"role": "assistant", "content": "{'Location': ['Morocco']}"}]
messages.extend(get_message("It's the Beijing Guoan team's turn this time, I wonder if they will follow suit?","{'organization': ['Beijing Guoan team']}"))
messages.extend(get_message("Revolutionary Sun Yat-sen established a branch of the Tongmenghui in Macau", "{'person's name': ['Sun Zhongshan'], 'place name': ['Macao'], 'organization': ['Tongmenghui']} "))
messages.extend(get_message("I used to work in Anhui Wuhu and Shanghai Pudong.","{'location': ['Anhui Wuhu', 'Shanghai Pudong']}"))
# display(messages)

def predict(text,temperature=0.01):
    model.generation_config.temperature=temperature
    response = model.chat(tokenizer, 
                          messages = messages+[{'role':'user','content':f'{text} -> '}])
    return response
print("Sample prediction", predict('Du Fu was a fan of Li Bai.'))

# %%
