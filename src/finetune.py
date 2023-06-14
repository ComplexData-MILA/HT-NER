import os, argparse, wandb, evaluate
import numpy as np
from dataset import loadDataset, ROOTS

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_SILENT"] = "true"
os.environ["TRANSFORMERS_CACHE"] = os.getenv("SCRATCH")

from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)

### Receive Augmentation
parser = argparse.ArgumentParser()
parser.add_argument("--base-model", type=str, required=True)
parser.add_argument("--datasets", nargs="+", type=str, required=True)
parser.add_argument("--only-loc", type=int, default=0)
parser.add_argument("--fold", type=int, default=-1)
parser.add_argument("--sub-structure", type=str, default="")
parser.add_argument("--substitude", type=int, default=0)
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

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
    ROOTS[dataset_name],
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

tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint, use_fast=True, add_prefix_space=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(
        "tokenizer.pad_token is None, set to tokenizer.eos_token: {}".format(
            tokenizer.eos_token
        )
    )
padding_value = 0 if "GP" in sub_structure else -100


def tokenize_and_align_labels(examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=512
    )

    labels = []
    for i, label in enumerate(examples[label_col_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(padding_value)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else padding_value)
            previous_word_idx = word_idx

        label_ids = [l2id[x] if type(x) != int else x for x in label_ids]
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
do_eval = "validation" in tokenized_datasets

config = AutoConfig.from_pretrained(
    model_checkpoint, id2label=id2l, label2id=l2id, num_labels=len(label_list)
)
# id2label (Dict[int, str], optional) — A map from index (for instance prediction index, or target index) to label.
# label2id (Dict[str, int], optional) — A map from label to index for the model.
# num_labels (int, optional) — Number of labels to use in the last layer added to the model, typically for a classification task.

structure_improve = args.sub_structure.split("-")
Deberta_ModelZoo = {
    "other": (AutoModelForTokenClassification, DataCollatorForTokenClassification),
}
ModelClass = Deberta_ModelZoo["other"]
config.BiLSTM = "BiL" in structure_improve
model = ModelClass[0].from_pretrained(
    model_checkpoint, ignore_mismatched_sizes=True, config=config
)

data_collator = ModelClass[1](tokenizer)
data_collator.num_labels = len(label_list)

metric = evaluate.load("seqeval")

model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    output_dir="./saved_models/"
    + f"{model_checkpoint.split('/')[-1]}{sub_structure}-{dataset_name}",
    overwrite_output_dir=True,
    seed=57706989,
    evaluation_strategy="epoch" if do_eval else "no",
    # evaluation_strategy = "steps",
    # eval_steps = 200, #14041 // 2 // batch_size,
    logging_steps=50,  # 14041 // 2 // batch_size,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 4,
    num_train_epochs=5,
    weight_decay=0.01,
    report_to="wandb",
    save_strategy="epoch",
    save_total_limit=3,
    # save_strategy='steps',
    # save_steps = 1200//4,
    # save_total_limit = 10
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    dataloader_num_workers=3 if "GP" not in structure_improve else 0,
    dataloader_pin_memory="GP" not in structure_improve,
    # metric_for_best_model="eval_f1",
    # greater_is_better=True,
    # load_best_model_at_end=True,
    local_rank=args.local_rank,
)

import torch
from transformers import Trainer
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    nested_detach,
    ALL_LAYERNORM_LAYERS,
)
from transformers.trainer_pt_utils import get_parameter_names


class NewTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
            if len(outputs) == 3:
                # loss, logits, labels = outputs
                return outputs
            if isinstance(outputs, TokenClassifierOutput):
                return (outputs.loss, outputs.logits, inputs.get("labels", None))
            return (outputs.loss, outputs.logits, None)

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            faster_parameters = ["classifier", "crf", "lstm", "gru", "global_pointer"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if n in decay_parameters
                        and not any(nn in n.lower() for nn in faster_parameters)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": 2e-5,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if n not in decay_parameters
                        and not any(nn in n.lower() for nn in faster_parameters)
                    ],
                    "weight_decay": 0.0,
                    "lr": 2e-5,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if any(nn in n.lower() for nn in faster_parameters)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": 1.5e-4,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
        return self.optimizer


def compute_metrics(p):
    predictions, labels = p
    labels = tokenized_datasets["validation"]["labels"]
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


trainer = NewTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"] if do_eval else None,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    # callbacks=[EarlyStoppingCallback()]
)

trainer.train(resume_from_checkpoint=False)
