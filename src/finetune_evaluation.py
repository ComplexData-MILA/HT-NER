from typing import List
import argparse, evaluate, os
from dataset import loadDataset, ROOTS
import numpy as np
import pandas as pd
import swifter

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from models.debertav2 import (
    DebertaV2CRF,
    DebertaV2GlobalPointer,
    DebertaV2TokenClassification,
)
from models.debertav2 import (
    DebertaV2CRFDataCollator,
    DebertaV2GlobalPointerDataCollator,
)


def main(args):
    model_checkpoint = args.base_model
    sub_structure = ""
    sub_structure += "-LOC" if args.only_loc else ""
    sub_structure += "" if args.fold == -1 else "-fold" + str(args.fold)
    sub_structure += "-" + args.sub_structure

    dataset_name = args.dataset
    batch_size = 100
    lr = 2e-5

    datasets, label_list, label_col_name = loadDataset(
        dataset_name,
        ROOTS[dataset_name],
        onlyLoc=args.only_loc,
        substitude=args.substitude,
    )
    l2id = {x: i for i, x in enumerate(label_list)}
    id2l = {i: x for i, x in enumerate(label_list)}
    print(label_list)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, use_fast=True, add_prefix_space=True
    )
    config = AutoConfig.from_pretrained(model_checkpoint)
    structure_improve = args.sub_structure.split("-")
    # only possible substructure: CRF, GP, BiL
    Deberta_ModelZoo = {
        "GP": (DebertaV2GlobalPointer, DebertaV2GlobalPointerDataCollator),
        "CRF": (DebertaV2CRF, DebertaV2CRFDataCollator),
        "BiL": (DebertaV2TokenClassification, DataCollatorForTokenClassification),
        "other": (AutoModelForTokenClassification, DataCollatorForTokenClassification),
    }
    assert not (
        "CRF" in structure_improve and "GP" in structure_improve
    ), "CRF and GP can only use one of them!"
    if "CRF" in structure_improve:
        ModelClass = Deberta_ModelZoo["CRF"]
    elif "GP" in structure_improve:
        ModelClass = Deberta_ModelZoo["GP"]
    elif "BiL" in structure_improve:
        ModelClass = Deberta_ModelZoo["BiL"]
    else:
        ModelClass = Deberta_ModelZoo["other"]
    config.BiLSTM = "BiL" in structure_improve

    model = ModelClass[0].from_pretrained(
        model_checkpoint, ignore_mismatched_sizes=True, config=config
    )

    data_collator = ModelClass[1](tokenizer)
    data_collator.num_labels = len(label_list)
    data_collator.max_length = 512

    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        output_dir="./saved_models/",  # +f"{model_checkpoint.split('/')[-1]}-{dataset_name}",
        overwrite_output_dir=False,
        seed=57706989,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 4,
        dataloader_num_workers=5 if "GP" not in structure_improve else 0,
        dataloader_pin_memory="GP" not in structure_improve,
        resume_from_checkpoint=args.base_model,
    )

    from metrics import f1

    # from tqdm import tqdm
    # tqdm.pandas()
    def evaluateHT(extractor, df, text_col="text", label_col="name"):
        assert label_col in df.columns
        pred_col = "pred"
        target_entity_map = {
            "name": ["PER", "NAME", "person"],
            "label": ["PER", "NAME", "person"],
            "location": ["LOC", "building", "location"],
        }
        name_set = target_entity_map[label_col]
        name_set = set(
            ["I-" + x for x in name_set] + ["B-" + x for x in name_set] + name_set
        )
        print(name_set)

        def step(row):
            text = row[text_col]
            pred = extractor(text)
            pred = "".join(p["word"] for p in pred if p["entity"] in name_set)
            pred = "|".join(filter(lambda x: x, pred.split("▁")))
            return pred

        df[pred_col] = df.swifter.apply(step, axis=1)
        df = df[[label_col, pred_col]]
        print(df.head())

        f1(df, df, [label_col], [pred_col])
        return df
    
    # print(trainer.evaluate(tokenized_datasets["validation"]))
    from preprocess.human_trafficking import getHTNameRaw, getHTUnifiedRaw
    from transformers import pipeline, TokenClassificationPipeline

    extractor = pipeline(
        model=model, tokenizer=tokenizer, task="ner", device=0
    )  # , aggregation_strategy = "simple"
    print("Evalute on HTName:")
    evaluateHT(extractor, getHTNameRaw(), label_col="label").to_csv(f"./results/htname_{dataset_name}_deberta.csv")

    print("Evalute on HTUnified:")
    evaluateHT(extractor, getHTUnifiedRaw(), label_col="name").to_csv(f"./results/htunified_name_{dataset_name}_deberta.csv")
    
    print("Evalute on HTUnified:")
    evaluateHT(extractor, getHTUnifiedRaw(), label_col="location").to_csv(f"./results/htunified_location_{dataset_name}_deberta.csv")
    # >>> token_classifier = pipeline(model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple")
    # >>> sentence = "Je m'appelle jean-baptiste et je vis à montréal"
    # >>> tokens = token_classifier(sentence)
    # >>> tokens
    # [{'entity': 'B-location',
    # 'score': 0.42658573,
    # 'index': 2,
    # 'word': 'golden',
    # 'start': 4,
    # 'end': 10},
    # {'entity': 'I-location',
    # 'score': 0.35856336,
    # 'index': 3,
    # 'word': 'state',
    # 'start': 11,
    # 'end': 16},
    # {'entity': 'B-group',
    # 'score': 0.3064001,
    # 'index': 4,
    # 'word': 'warriors',
    # 'start': 17,
    # 'end': 25},
    # {'entity': 'B-location',
    # 'score': 0.65523505,
    # 'index': 13,
    # 'word': 'san',
    # 'start': 80,
    # 'end': 83},
    # {'entity': 'B-location',
    # 'score': 0.4668663,
    # 'index': 14,
    # 'word': 'francisco',
    # 'start': 84,
    # 'end': 93}]


if __name__ == "__main__":
    ### Receive Augmentation
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--only-loc", type=int, default=0)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--sub-structure", type=str, default="")
    parser.add_argument("--substitude", type=int, default=0)
    args = parser.parse_args()
    main(args)