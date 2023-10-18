from typing import List
import argparse, evaluate, os, re
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


def main(args):
    model_checkpoint = args.base_model
    sub_structure = ""
    sub_structure += "-LOC" if args.only_loc else ""
    sub_structure += "" if args.fold == -1 else "-fold" + str(args.fold)
    sub_structure += "-" + args.sub_structure

    dataset_name = args.datasets  # [0]
    batch_size = 50
    # lr = 2e-5

    # datasets, label_list, label_col_name = loadDataset(
    #     dataset_name,
    #     ROOTS[dataset_name],
    #     onlyLoc=args.only_loc,
    #     substitude=args.substitude,
    # )
    # l2id = {x: i for i, x in enumerate(label_list)}
    # id2l = {i: x for i, x in enumerate(label_list)}
    # print(label_list)

    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, use_fast=True, add_prefix_space=True
    )
    config = AutoConfig.from_pretrained(model_checkpoint)
    structure_improve = args.sub_structure.split("-")
    # only possible substructure: CRF, GP, BiL
    Deberta_ModelZoo = {
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
    data_collator.num_labels = len(config.id2label)  # len(label_list)
    data_collator.max_length = 512

    model_name = model_checkpoint.split("/")[-2]
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

    from neat_metrics import f1
    from itertools import chain
    from transformers import pipeline, TokenClassificationPipeline
    from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok

    def evaluateHT(extractor, df, text_col="text", label_col="name"):
        assert label_col in df.columns
        pred_col = "pred"
        target_entity_map = {
            "name": ["PER", "NAME", "person", "PERSON"],
            "gpt_name": ["PER", "NAME", "person", "PERSON"],
            "label": ["PER", "NAME", "person", "PERSON"],
            "location": ["LOC", "building", "location", "GPE"],
            "gpt_location": ["LOC", "building", "location", "GPE"],
        }
        name_set = target_entity_map[label_col]
        name_set = set(
            ["I-" + x for x in name_set] + ["B-" + x for x in name_set] + name_set
        )
        print(name_set)

        def step(row):
            text = row[text_col]
            pred = extractor(text)
            pred = [
                x for x in pred if x.get("entity", x.get("entity_group")) in name_set
            ]
            new = []
            for i, x in enumerate(pred):
                if i == 0:
                    new.append(x)
                else:
                    if x["start"] == new[-1]["end"]:
                        new[-1]["end"] = x["end"]
                        new[-1]["word"] += x["word"]
                    # elif x["start"] == new[-1]["end"] + 1:
                    #     new[-1]["end"] = x["end"]
                    #     new[-1]["word"] += " " + x["word"]
                    else:
                        new.append(x)
            return "|".join([x["word"].strip() for x in new])

            # return "|".join([x["word"] for x in pred])
            # return "|".join(filter(None, "".join([x["word"] for x in pred]).split()))

            # pred = [x for x in pred if x["entity"] in name_set]

            # # connect tokens, if start index and next end index is the same, then connect
            # new = []
            # for i, x in enumerate(pred):
            #     if i == 0:
            #         new.append(x)
            #     else:
            #         if x["start"] == new[-1]["end"]:
            #             new[-1]["end"] = x["end"]
            #             new[-1]["word"] += x["word"]
            #         elif x["start"] == new[-1]["end"] + 1:
            #             new[-1]["end"] = x["end"]
            #             new[-1]["word"] += " " + x["word"]
            #         else:
            #             new.append(x)

            # return "|".join([x["word"] for x in new]).replace("Ġ", "").replace("▁", "")
            # return "|".join(filter(None, "".join([x["word"] for x in pred]).split()))

        df[pred_col] = df.swifter.apply(step, axis=1)
        df = df[[label_col, pred_col]]
        print(df.head())

        f1(df, df, [label_col], [pred_col])
        return df

    extractor = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="ner",
        device=0,
        aggregation_strategy="simple",
    )

    from preprocess.human_trafficking import (
        getHTNameRaw,
        getHTUnifiedRaw,
        getHTGen6kRaw,
        getHTGen12kRaw,
        getHTGenV2Raw,
    )

    print("Evalute on HTName:")
    HTUName = evaluateHT(extractor, getHTNameRaw(), label_col="label")
    # HTULocation = evaluateHT(extractor, getHTNameRaw(), label_col="location")

    HTUName.rename(columns={"pred": "name"}, inplace=True)
    # HTUName["location"] = HTULocation["pred"]
    HTUName.to_csv(f"./results/ht_dataset/HTName_{model_name}.csv", index=False)

    print("Evalute on HTUnified:")
    HTUName = evaluateHT(extractor, getHTUnifiedRaw(), label_col="name")
    HTULocation = evaluateHT(extractor, getHTUnifiedRaw(), label_col="location")

    HTUName.rename(columns={"pred": "name"}, inplace=True)
    HTUName["location"] = HTULocation["pred"]
    HTUName.to_csv(f"./results/ht_dataset/HTUnified_{model_name}.csv", index=False)

    print("Evalute on HTGen 6k:")
    HTUName = evaluateHT(extractor, getHTGen6kRaw(), label_col="gpt_name")
    HTUName.to_csv(f"./results/ht_dataset/HTGen6k_{model_name}.csv", index=False)

    print("Evalute on HTGen 12K:")
    HTUName = evaluateHT(extractor, getHTGen12kRaw(), label_col="gpt_name")
    HTUName.to_csv(f"./results/ht_dataset/HTGen12k_{model_name}.csv", index=False)

    print("Evalute on HTGen V2:")
    HTUName = evaluateHT(extractor, getHTGenV2Raw(), label_col="gpt_name")
    HTUName.to_csv(f"./results/ht_dataset/HTGenV2_{model_name}.csv", index=False)


if __name__ == "__main__":
    ### Receive Augmentation
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str)
    parser.add_argument("--datasets", type=str)
    parser.add_argument("--only-loc", type=int, default=0)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--sub-structure", type=str, default="")
    parser.add_argument("--substitude", type=int, default=0)
    args = parser.parse_args()
    main(args)
