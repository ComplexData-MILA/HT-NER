import pandas as pd
import os
from os.path import join as pj
from nltk.tokenize import RegexpTokenizer

cache_path = "data/cache/HT/"
os.makedirs(cache_path, exist_ok=True)
delEmpty = lambda x: [] if all(not y for y in x) else x
tokenizer = RegexpTokenizer(r"[/|(|)|{|}|$|?|!]|\w+|\$[\d\.]+|\S+")
name_tokenizer = RegexpTokenizer(r"\w+")
location_tokenizer = RegexpTokenizer(r"\w+|\$[\d\.]+")


def parse_name(n, default):
    if "loc" in n.lower():
        return "LOC"
    elif "name" in n.lower():
        return "NAME"
    return default


def load(fn, text_cols, label_cols, default, tokenize=True):

    df = pd.read_csv(fn)
    df = df[text_cols + label_cols]

    df.fillna("", inplace=True)
    df["text"] = df.apply(lambda x: " ".join(x[col] for col in text_cols), axis=1)

    df["text"] = df["text"].apply(lambda x: x.replace("’", "'"))
    df[label_cols] = df[label_cols].apply(lambda x: x.replace("’", "'"))

    if not tokenize:
        return df[["text"] + label_cols]

    tokens, tags = [], []
    for i, line in df.iterrows():
        text = line["text"]
        token = tokenizer.tokenize(text)
        tokens.append(token)

        tag = ["O" for _ in range(len(token))]
        for label_col in label_cols:
            tag_name = parse_name(label_col, default)

            if tag_name == "NAME":
                tag = update_label_list4name(token, line[label_col], tag, tag_name)
                # print(tag_name, line[label_col], label_col)
            elif "unsup" in fn.lower():
                tag = update_label_list4loc_unsup(token, line[label_col], tag, tag_name)
                # print(tag_name, line[label_col], label_col)
            else:
                tag = update_label_list4loc(token, line[label_col], tag, tag_name)

        tags.append(tag)

    return pd.DataFrame({"tokens": tokens, "tags": tags})


def update_label_list4name(tokenized_text, label, label_list, label_name):

    # init
    pointer = 0
    tmp_token = tokenized_text[::]
    tmp_token_lower = [t.lower() for t in tmp_token]
    ts, tls = set(tmp_token), set(tmp_token_lower)
    if ";" in label:
        label = label.replace(";", "|")
    label_splited = [x.strip() for x in delEmpty(label.split("|"))]

    for tag_single in label_splited:
        for ind, ttag in enumerate(name_tokenizer.tokenize(tag_single)):
            # if ttag in ts:
            #     s = tmp_token.index(ttag)
            # elif ttag.lower() in tls:
            #     try:
            #         s = tmp_token_lower.index(ttag.lower())
            #     except:
            #         print(tmp_token_lower)
            #         print(ttag.lower())
            #         exit()
            all_positions = []
            for i, t in enumerate(tmp_token_lower):
                if ttag.lower() in t:
                    all_positions.append(i)
                    break

            if len(all_positions) == 0:
                # print(tmp_token)
                # print(ttag)
                print(label_name, ttag)
                continue
                # raise Exception("Not found")

            for pos in all_positions:
                if ind == 0:
                    label_list[pointer + pos] = f"B-{label_name}"
                else:
                    label_list[pointer + pos] = f"I-{label_name}"

    return label_list


def update_label_list4loc(tokenized_text, label, label_list, label_name):

    # init
    pointer = 0
    tmp_token = tokenized_text[::]
    label_splited = [x.strip() for x in delEmpty(label.split("|"))]

    for tag_single in label_splited:
        for ind, ttag in enumerate(location_tokenizer.tokenize(tag_single)):
            s = tmp_token.index(ttag)
            if ind == 0:
                label_list[pointer + s] = f"B-{label_name}"
            else:
                label_list[pointer + s] = f"I-{label_name}"
            pointer += s
            tmp_token = tmp_token[s:]

    return label_list


def update_label_list4loc_unsup(tokenized_text, label, label_list, label_name):
    words_trash = set(
        [
            "miami",
            "house",
            "home",
            "apartment",
            "hotel",
            "room",
            "bedroom",
            "bed",
            "bathroom",
            "bath",
            "kitchen",
            "living",
            "downtown",
            "my",
            "place",
            "location",
            "neighborhood",
            "city",
            "town",
            "village",
            "suburb",
            "suburban",
        ]
    )
    # init
    pointer = 0
    tmp_token = tokenized_text[::]
    tmp_token_lower = [t.lower() for t in tmp_token]
    ts, tls = set(tmp_token), set(tmp_token_lower)
    label_splited = [x.strip() for x in delEmpty(label.split("|"))]

    for tag_single in label_splited:
        if len(tag_single) <= 1:
            continue
        for ind, ttag in enumerate(location_tokenizer.tokenize(tag_single)):
            if ttag in words_trash:
                continue
            all_positions = []
            for i, t in enumerate(tmp_token_lower):
                if ttag.lower() in t:
                    all_positions.append(i)
                    break

            if len(all_positions) == 0:
                # print(tmp_token)
                print(tag_single, "|", ttag)
                continue
                # raise Exception("Not found")

            for pos in all_positions:
                if ind == 0:
                    label_list[pointer + pos] = f"B-{label_name}"
                else:
                    label_list[pointer + pos] = f"I-{label_name}"

    return label_list


def getHTNameRaw():
    return load(
        "data/HT/HTName.csv", ["title", "description"], ["label"], "NAME", False
    )


def getHTUnifiedRaw():
    return load(
        "data/HT/HTUnified.csv",
        ["title", "description"],
        ["location", "name"],
        "NAME",
        False,
    )


def getHTUnsupRaw():
    return load(
        "results/HTUnsup_chatgpt.csv",
        ["title", "description"],
        ["gpt_location", "gpt_name"],
        "NAME",
        False,
    )


if __name__ == "__main__":
    # Name
    df = load(
        "data/HT/HTName.csv",
        ["title", "description"],
        ["label"],
        "NAME",
    )

    df.to_csv(pj(cache_path, "HTName_tokenized.csv"), index=False)

    # Unified
    df = load(
        "data/HT/HTUnified.csv",
        ["title", "description"],
        ["location", "name"],
        "NAME",
    )

    df.to_csv(pj(cache_path, "HTUnified_tokenized.csv"), index=False)

    # Unsup
    df = load(
        "results/HTUnsup_chatgpt.csv",
        ["title", "description"],
        ["gpt_location", "gpt_name"],
        "NAME",
    )
    # name: 9 miss matched

    df.to_csv(pj(cache_path, "HTUnsup_tokenized.csv"), index=False)
