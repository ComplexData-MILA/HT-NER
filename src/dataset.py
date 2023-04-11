# %%
from functools import partial
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import pandas as pd
import os
from os.path import join as pj

# datasets = load_dataset("conll2003")
# datasets = load_dataset("tner/wnut2017")


def load_street_name(root="../data/oda"):
    import json
    from glob import glob

    all_data = {}
    state_short = ["AB", "BC", "MB", "NB", "NS", "NT", "ON", "PE", "QC", "SK"]
    for state in state_short:
        with open(
            pj(root, f"{state}_processed_streets.json"), "r", encoding="utf-16"
        ) as f:
            content = json.load(f)
            all_data.update(content)
    print("Finish Loading Street Names from CA Open Source.")
    return all_data


def _loadWrapper(ds_name, root, kargs):
    if ds_name in ["conll2003", "conll-2003"]:
        return conll2003()
    elif ds_name in ["wnut2017"]:
        return wnut2017()
    elif ds_name in ["fewnerd", "few-nerd"]:
        assert root, "fewnerd need root path"
        return fewnerd(root)
    elif ds_name in ["fewnerd-l1", "few-nerd-l1"]:
        assert root, "fewnerd need root path"
        return fewnerd(root, True)
    elif ds_name in ["fewnerd-only", "few-nerd-onlyi"]:
        assert root, "fewnerd need root path"
        return fewnerd_onlyI(root)
    elif ds_name in ["fewnerd-l1-onlyi", "few-nerd-l1-onlyi"]:
        assert root, "fewnerd need root path"
        return fewnerd_onlyI(root, True)
    elif ds_name in ["wikiner", "wikiner-en", "wikineren"]:
        assert root, "wikiner need root path"
        return wikiner(root)
    elif ds_name in ["HTName", "HTname", "htname"]:
        assert root, "HT dataset need root path"
        return ht_name(root, kargs)
    elif ds_name in ["HTUnified", "HTunified", "htunified"]:
        assert root, "HT dataset need root path"
        return ht_unified(root, kargs)
    elif ds_name in ["HTUnsup", "HTunsup", "htunsup"]:
        assert root, "HT dataset need root path"
        return ht_unsup(root, kargs)
    else:
        assert False, "Error, Please check names of datasets!"


def loadDataset(ds_name, root="", unique="", substitude=False, fold=-1, **kargs):
    print("Loading Dataset: ", ds_name)
    # by default, num of folds = 5, if fold == -1, not fold, otherwise, fold/5 th fold
    ds_name = ds_name.lower()
    assert ds_name in [
        "conll2003",
        "fewnerd",
        "few-nerd",
        "fewnerd-l1",
        "few-nerd-l1",
        "fewnerd-l1-onlyi",
        "few-nerd-l1-onlyi",
        "wnut2017",
        "wikiner",
        "wikiner-en",
        "wikineren",
        "HTName",
        "HTUnified",
        "HTUnsup",
        "htname",
        "htunified",
        "htunsup",
        "HTname",
        "HTlocation",
        "HTunified",
    ]
    ds, label_list, label_col_name = _loadWrapper(ds_name, root, kargs)
    # postLoadDataset(ds, label_list, label_col_name)
    # only LOC

    def map_fn(examples, B="B-LOC", I="I-LOC", k="loc", col_name="tags"):
        new_label = []
        for l in examples[col_name]:
            if type(l[0]) == int:
                new_label.append(
                    [
                        (I if "I-" in label_list[ll] else B)
                        if k in label_list[ll].lower()
                        else "O"
                        for ll in l
                    ]
                )
            else:
                new_label.append(
                    [(I if "I-" in ll else B) if k in ll.lower() else "O" for ll in l]
                )
        examples[col_name] = new_label
        return examples

    if unique == "Location":
        label_list = ["O", "B-LOC", "I-LOC"]
        ds = ds.map(
            partial(map_fn, B="B-LOC", I="I-LOC", k="loc", col_name=label_col_name),
            batched=True,
        )
    elif unique == "Name":
        label_list = ["O", "B-NAME", "I-NAME"]
        ds = ds.map(
            partial(map_fn, B="B-NAME", I="I-NAME", k="name", col_name=label_col_name),
            batched=True,
        )
    elif unique == "":
        pass
    else:
        assert False, "Error, Please check [unique]!"

    if substitude:
        street_data = load_street_name()  # {'city': list [street name]}
        # street_data_classified_by_length = {i:[street for city in street_data.values() for street in city if (street)==i] for i in range(1, 10)}
        street_list = [
            street for city in street_data.values() for street in city
        ] + list(street_data.keys()) * 10
        street_list = list(set(street_list))
        street_list.sort()
        loc_tag = [l for l in label_list if "loc" in l.lower()]

        # import matplotlib.pyplot as plt
        # plt.hist(list(map(len, street_list)), bins=20)
        # plt.show()
        # plt.savefig('tmp.png')
        # print(len(street_list))

        import random

        random.seed(2022)
        random.shuffle(street_list)
        from nltk.tokenize import RegexpTokenizer

        tokenizer = RegexpTokenizer(r"[/|(|)|{|}|$|?|!]|\w+|\$[\d\.]+|\S+")

        def f(examples):
            new_label = []
            new_token = []
            from itertools import groupby

            for t, l in zip(examples["tokens"], examples[label_col_name]):
                l = [label_list[x] for x in l] if type(l[0]) == int else l

                groups = [[[], []]]
                for i, (tt, ll) in enumerate(zip(t, l)):
                    if (ll in loc_tag) != (l[i - 1] in loc_tag):
                        groups.append([[], []])
                    groups[-1][0].append(tt)
                    groups[-1][1].append(ll)

                new_token.append([])
                new_label.append([])
                # print(groups)
                for tt, ll in groups:
                    if not (tt or ll):
                        continue
                    if ll[0] in loc_tag:
                        new_string = random.choice(street_list)
                        tt = tokenizer.tokenize(new_string)
                        if len(ll) == 1 and len(tt) > 1:
                            ll += [ll[0].replace("B-", "I-")] * (len(tt) - 1)
                        else:
                            ll = [ll[0]] + [ll[-1]] * (len(tt) - 1)
                        new_token[-1].extend(tt)
                        new_label[-1].extend(ll)
                    else:
                        new_token[-1].extend(tt)
                        new_label[-1].extend(ll)

            examples["newtags"] = new_label
            examples["tokens"] = new_token
            return examples

        ds = ds.map(f, batched=True)
        label_col_name = "newtags"

    if fold != -1:
        assert fold >= 5 or fold < -1
        from sklearn.model_selection import StratifiedKFold
        import numpy as np

        folds = StratifiedKFold(n_splits=5)
        splits = folds.split(
            np.zeros(ds["train"].num_rows), ds["train"][label_col_name]
        )
        for i, (train_idxs, val_idxs) in enumerate(splits):
            if i == fold:
                ds["test"] = ds["validation"]
                ds["validation"] = ds["train"].select(val_idxs)
                ds["train"] = ds["train"].select(train_idxs)
                print("KFold Update Correctly!")

    return ds, label_list, label_col_name


# def postLoadDataset(ds, label_list, label_col_name)


def conll2003(root=""):
    datasets = load_dataset("conll2003")
    # {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    label_list = [
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-MISC",
        "I-MISC",
    ]
    return datasets, label_list, "ner_tags"


def wnut2017(root=""):
    label_list = [
        "O",
        "B-corporation",
        "I-corporation",
        "B-creative-work",
        "I-creative-work",
        "B-group",
        "I-group",
        "B-location",
        "I-location",
        "B-person",
        "I-person",
        "B-product",
        "I-product",
    ]
    # 1: B-corporation
    # 2: I-corporation
    # 3: B-creative-work
    # 4: I-creative-work
    # 5: B-group
    # 6: I-group
    # 7: B-location
    # 8: I-location
    # 9: B-person
    # 10: I-person
    # 11: B-product
    # 12: I-product
    return load_dataset("wnut_17"), label_list, "ner_tags"


def fewnerd_onlyI(root, only_l1=False):
    if (only_l1 and not os.path.exists(pj(root, "train_L1.csv"))) or (
        not only_l1 and not os.path.exists(pj(root, "train.csv"))
    ):
        data_path = "./data/Few-NERD/"
        # cache_path = "./data/cache/Few-NERD/"
        os.makedirs(root, exist_ok=True)
        for file in ["train.txt", "test.txt", "dev.txt"]:
            with open(pj(data_path, file), "r", encoding="utf-8") as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]

            texts, tags, tags_l1 = [[]], [[]], [[]]

            for line in lines:
                if line:
                    text, tag = line.split("\t")
                    texts[-1].append(text)
                    tags[-1].append(tag)
                    tags_l1[-1].append(tag.split("-")[0])
                else:
                    texts.append([])
                    tags.append([])
                    tags_l1.append([])

            pd.DataFrame({"tokens": texts, "tags": tags}).to_csv(
                pj(root, file.replace(".txt", ".csv")), index=False
            )
            pd.DataFrame({"tokens": texts, "tags": tags_l1}).to_csv(
                pj(root, file.replace(".txt", "_L1.csv")), index=False
            )


    tds = Dataset.from_pandas(
        help_load(
            pd.read_csv(pj(root, "train_L1.csv" if only_l1 else "train.csv")), toIO
        )
    )
    vds = Dataset.from_pandas(
        help_load(pd.read_csv(pj(root, "test_L1.csv" if only_l1 else "test.csv")), toIO)
    )
    devds = Dataset.from_pandas(
        help_load(pd.read_csv(pj(root, "dev_L1.csv" if only_l1 else "dev.csv")), toIO)
    )

    datasets = DatasetDict()

    datasets["train"] = tds
    datasets["validation"] = vds
    datasets["test"] = devds

    label_list = [  # 76
        "art-broadcastprogram",
        "art-film",
        "art-music",
        "art-artother",
        "art-painting",
        "art-writtenart",
        "building-airport",
        "building-hospital",
        "building-hotel",
        "building-library",
        "building-buildingother",
        "building-restaurant",
        "building-sportsfacility",
        "building-theater",
        "event-attack/battle/war/militaryconflict",
        "event-disaster",
        "event-election",
        "event-eventother",
        "event-protest",
        "event-sportsevent",
        "location-GPE",  # 20
        "location-bodiesofwater",
        "location-island",
        "location-mountain",
        "location-locationother",
        "location-park",
        "location-road/railway/highway/transit",  # 26
        "organization-company",
        "organization-education",
        "organization-government/governmentagency",
        "organization-media/newspaper",
        "organization-organizationother",
        "organization-politicalparty",
        "organization-religion",
        "organization-showorganization",
        "organization-sportsleague",
        "organization-sportsteam",
        "other-astronomything",
        "other-award",
        "other-biologything",
        "other-chemicalthing",
        "other-currency",
        "other-disease",
        "other-educationaldegree",
        "other-god",
        "other-language",
        "other-law",
        "other-livingthing",
        "other-medical",
        "person-actor",
        "person-artist/author",
        "person-athlete",
        "person-director",
        "person-personother",
        "person-politician",
        "person-scholar",
        "person-soldier",
        "product-airplane",
        "product-car",
        "product-food",
        "product-game",
        "product-productother",
        "product-ship",
        "product-software",
        "product-train",
        "product-weapon",
    ]
    label_list = ["O"] + ["I-" + x for x in label_list]

    if only_l1:
        label_list = [
            "art",  # 1,9
            "building",  # 2,10
            "event",
            "location",  # 3, 12
            "organization",  # 5
            "other",
            "person",
            "product",  # 8
        ]
        label_list = ["O"] + ["I-" + x for x in label_list]

    return datasets, label_list, "tags"


def fewnerd(root, only_l1=False):
    if (only_l1 and not os.path.exists(pj(root, "train_L1.csv"))) or (
        not only_l1 and not os.path.exists(pj(root, "train.csv"))
    ):

        data_path = "./data/Few-NERD/"
        # cache_path = "./data/cache/Few-NERD/"
        os.makedirs(root, exist_ok=True)
        for file in ["train.txt", "test.txt", "dev.txt"]:
            with open(pj(data_path, file), "r", encoding="utf-8") as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]

            texts, tags, tags_l1 = [[]], [[]], [[]]

            for line in lines:
                if line:
                    text, tag = line.split("\t")
                    texts[-1].append(text)
                    tags[-1].append(tag)
                    tags_l1[-1].append(tag.split("-")[0])
                else:
                    texts.append([])
                    tags.append([])
                    tags_l1.append([])

            pd.DataFrame({"tokens": texts, "tags": tags}).to_csv(
                pj(root, file.replace(".txt", ".csv")), index=False
            )
            pd.DataFrame({"tokens": texts, "tags": tags_l1}).to_csv(
                pj(root, file.replace(".txt", "_L1.csv")), index=False
            )

    tds = Dataset.from_pandas(
        help_load(
            pd.read_csv(pj(root, "train_L1.csv" if only_l1 else "train.csv")), toBIO
        )
    )
    vds = Dataset.from_pandas(
        help_load(
            pd.read_csv(pj(root, "test_L1.csv" if only_l1 else "test.csv")), toBIO
        )
    )
    devds = Dataset.from_pandas(
        help_load(pd.read_csv(pj(root, "dev_L1.csv" if only_l1 else "dev.csv")), toBIO)
    )

    datasets = DatasetDict()

    datasets["train"] = tds
    datasets["validation"] = vds
    datasets["test"] = devds

    label_list = [  # 76
        "art-broadcastprogram",
        "art-film",
        "art-music",
        "art-artother",
        "art-painting",
        "art-writtenart",
        "building-airport",
        "building-hospital",
        "building-hotel",
        "building-library",
        "building-buildingother",
        "building-restaurant",
        "building-sportsfacility",
        "building-theater",
        "event-attack/battle/war/militaryconflict",
        "event-disaster",
        "event-election",
        "event-eventother",
        "event-protest",
        "event-sportsevent",
        "location-GPE",  # 20
        "location-bodiesofwater",
        "location-island",
        "location-mountain",
        "location-locationother",
        "location-park",
        "location-road/railway/highway/transit",  # 26
        "organization-company",
        "organization-education",
        "organization-government/governmentagency",
        "organization-media/newspaper",
        "organization-organizationother",
        "organization-politicalparty",
        "organization-religion",
        "organization-showorganization",
        "organization-sportsleague",
        "organization-sportsteam",
        "other-astronomything",
        "other-award",
        "other-biologything",
        "other-chemicalthing",
        "other-currency",
        "other-disease",
        "other-educationaldegree",
        "other-god",
        "other-language",
        "other-law",
        "other-livingthing",
        "other-medical",
        "person-actor",
        "person-artist/author",
        "person-athlete",
        "person-director",
        "person-personother",
        "person-politician",
        "person-scholar",
        "person-soldier",
        "product-airplane",
        "product-car",
        "product-food",
        "product-game",
        "product-productother",
        "product-ship",
        "product-software",
        "product-train",
        "product-weapon",
    ]
    label_list = ["O"] + ["B-" + x for x in label_list] + ["I-" + x for x in label_list]

    if only_l1:
        label_list = [
            "art",  # 1,9
            "building",  # 2,10
            "event",
            "location",  # 3, 12
            "organization",  # 5
            "other",
            "person",
            "product",  # 8
        ]
        label_list = (
            ["O"] + ["B-" + x for x in label_list] + ["I-" + x for x in label_list]
        )

    return datasets, label_list, "tags"


def ontonotev5(root):
    # only english
    assert NotImplemented


def wikiner(root, language="en"):
    assert language in ["en"]
    import bz2
    import pandas as pd

    # with bz2.open("data/wikiner-en/aij-wikiner-en-wp2.bz2", "rb") as f:
    #     content = f.read().decode('utf-8').split('\n')

    # texts, tags = [], []
    # # print(content[:20])
    # for sent in content:
    #     if not sent: continue
    #     text, tag = [], []
    #     for word in sent.split(" "):
    #         word, _, label = word.split("|")
    #         text, tag = text+[word], tag+[label]
    #         # print(word, label)
    #     texts, tags = texts.append(text), tags.append(tag)

    # pd.DataFrame({'text': texts, 'tag': tags})
    full_df = help_load(pd.read_csv(pj(root, "aij-wikiner-en-wp2.csv")), toBIO)
    ratio = [0.9, 0.1]
    # ratio = {'train':int(0.8*full_ds), 'valid':0.1, 'test':0.1}
    tds = Dataset.from_pandas(full_df.iloc[0 : int(ratio[0] * len(full_df))])
    vds = Dataset.from_pandas(full_df.iloc[int(ratio[0] * len(full_df)) :])

    datasets = DatasetDict()

    datasets["train"] = tds
    datasets["validation"] = vds

    label_list = [
        "O",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-PER",
        "I-PER",
        "B-MISC",
        "I-MISC",
    ]
    # label_list = ['O', 'I-ORG', 'I-LOC', 'I-PER', 'I-MISC']

    return datasets, label_list, "tags"


def ht_name(root, kargs):
    import pandas as pd

    full_df = help_load(pd.read_csv(pj(root, "HTName_tokenized.csv")))

    # if kargs.get("filter_empty_label", False):
    #     contain_label = []
    #     for i, t in full_df.iterrows():
    #         if "B-LOC" in t["tags"]:
    #             contain_label.append(i)

    #     full_df = full_df.iloc[contain_label]

    tds = vds = Dataset.from_pandas(full_df)
    datasets = DatasetDict()
    datasets["train"] = tds
    datasets["validation"] = vds
    label_list = ["O", "B-NAME", "I-NAME"]
    return datasets, label_list, "tags"


def ht_unified(root, kargs):
    import pandas as pd

    full_df = help_load(pd.read_csv(pj(root, "HTUnified_tokenized.csv")))

    # if kargs.get("filter_empty_label", False):
    #     contain_label = []
    #     for i, t in full_df.iterrows():
    #         if "B-LOC" in t["tags"]:
    #             contain_label.append(i)
    #     full_df = full_df.iloc[contain_label]

    tds = vds = Dataset.from_pandas(full_df)
    datasets = DatasetDict()
    datasets["train"] = tds
    datasets["validation"] = vds
    label_list = ["O", "B-LOC", "I-LOC", "B-NAME", "I-NAME"]
    return datasets, label_list, "tags"


def ht_unsup(root, kargs):
    import pandas as pd

    full_df = help_load(pd.read_csv(pj(root, "HTUnsup_tokenized.csv")))

    # if kargs.get("filter_empty_label", False):
    #     contain_label = []
    #     ls = set(["B-LOC", "B-NAME"])
    #     for i, t in full_df.iterrows():
    #         if ls.intersection(t["tags"]):
    #             contain_label.append(i)
    #     full_df = full_df.iloc[contain_label]

    tds = vds = Dataset.from_pandas(full_df)
    datasets = DatasetDict()
    datasets["train"] = tds
    datasets["validation"] = vds
    label_list = ["O", "B-LOC", "I-LOC", "B-NAME", "I-NAME"]
    return datasets, label_list, "tags"


# utils function
from ast import literal_eval


def toBIO(x):
    if x:
        new = ["O"] * len(x)
        new[0] = "O" if x[0] == "O" else "B-" + x[0]
        for i in range(1, len(x)):
            if x[i] != "O":
                if x[i - 1] != x[i]:
                    new[i] = "B-" + x[i]
                else:
                    new[i] = "I-" + x[i]
        return new
    return x


def fixBIO(x):
    if x:
        new = ["O"] * len(x)
        new[0] = "O" if x[0] == "O" else x[0].replace("I-", "B-")
        for i in range(1, len(x)):
            if x[i] == "O":
                continue
            if x[i - 1] != x[i] and "B-" not in x[i]:
                new[i] = x[i].replace("I-", "B-")
            else:
                new[i] = x[i]
        return new
    return x


def toIO(x):
    if x:
        return ["O" if y == "O" else "I-" + y for y in x]
    return x


def help_load(df, f=None):
    df["tokens"] = df["tokens"].map(literal_eval)
    df["tags"] = df["tags"].map(literal_eval)
    if f is not None:
        df["tags"] = df["tags"].map(f)
    return df


# %%
if __name__ == "__main__":
    # data = load_street_name('/home/mila/h/hao.yu/ht/HTResearch/data/oda')
    # print(len(data))
    from pprint import pprint

    print(loadDataset("conll2003")[0]["train"]["tokens"][4])
    print(loadDataset("wnut2017")[0]["train"]["tokens"][4])

    print(
        loadDataset("fewnerd-l1", root="./data/cache/Few-NERD")[0]["train"]["tokens"][4]
    )
    print(
        loadDataset("wikiner-en", root="./data/cache/wikiner-en")[0]["train"][
            "tokens"
        ][4]
    )
    print(loadDataset("HTName", root="./data/cache/HT")[0]["train"]["tokens"][4])
    print(loadDataset("HTUnified", root="./data/cache/HT")[0]["train"]["tokens"][4])
    print(loadDataset("HTUnsup", root="./data/cache/HT")[0]["train"]["tokens"][4])