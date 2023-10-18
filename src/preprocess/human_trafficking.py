import pandas as pd
import os
from os.path import join as pj
from nltk.tokenize import RegexpTokenizer

cache_path = "data/cache/HT/"
os.makedirs(cache_path, exist_ok=True)
delEmpty = lambda x: [] if all(not y for y in x) else x
tokenizer = RegexpTokenizer(r"[/|(|)|{|}|$|?|!]|\w+|\$[\d\.]+|\S+")
name_tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
location_tokenizer = RegexpTokenizer(r"\w+|\$[\d\.|-]+|\d+")


def parse_name(n, default):
    if "loc" in n.lower():
        return "LOC"
    elif "name" in n.lower():
        return "NAME"
    return default


def load(fn, text_cols, label_cols, default, tokenize=True):
    print(f"Processing: {fn}")

    df = pd.read_csv(fn)
    df = df[text_cols + label_cols]

    df.fillna("", inplace=True)
    df["text"] = df.apply(lambda x: " ".join(x[col] for col in text_cols), axis=1)

    df["text"] = df["text"].apply(lambda x: x.replace("’", "'"))
    df[label_cols] = df[label_cols].apply(lambda x: x.replace("’", "'"))

    if "HTGen" in fn:
        texts = []

        def nf(line):
            text = line["text"]
            text = text.replace("Title: ", "").replace("Description: ", "")
            if len(text) < 20:
                return False
            filter_words = ["realistic", "examples", "sure", "human", "trafficking"]
            if any(x in text.lower() for x in filter_words):
                # print(text)
                if len(text.split(":")[-1]) > 2:
                    text = text.split(":")[-1]
                    if any(x in text.lower() for x in filter_words):
                        return False
                else:
                    return False
            if len(tokenizer.tokenize(text)) < 5:
                return False

            texts.append(text)
            return True

        # df[~df.apply(nf, axis=1)].to_csv("./tmp.csv", index=False)
        if "V2" not in fn:
            df = df[df.apply(nf, axis=1)]
            df["text"] = texts

        df = df[["text"] + label_cols]

        def wp(label):
            return "|".join(preprocess_label_text(label, True))

        df["gpt_name"] = df["gpt_name"].apply(wp)

    if not tokenize:
        return df[["text"] + label_cols]

    tokens, tags = [], []
    for i, line in df.iterrows():
        text = line["text"]
        token = tokenizer.tokenize(text)

        if len(token) < 5:
            continue

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


def preprocess_label_text(label, more=False):
    if ";" in label:
        label = label.replace(";", "|")
    if more:
        if len(label) > 25:
            label = label.split("||")[-1]
            if len(label) > 25:
                label = label.split('"|')[-1]
                if "1. " in label:
                    label = label.split("|")[-1]
                    label = "|".join(xl for xl in label.split("|") if len(xl) < 20)
    label_splited = [x.strip() for x in delEmpty(label.split("|"))]
    return label_splited


def update_label_list4name(tokenized_text, label, label_list, label_name):

    # init
    pointer = 0
    tmp_token = tokenized_text[::]
    tmp_token_lower = [t.lower() for t in tmp_token]
    ts, tls = set(tmp_token), set(tmp_token_lower)
    label_splited = preprocess_label_text(label)
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
                print(tmp_token)
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
            try:
                s = tmp_token.index(ttag)
                if ind == 0:
                    label_list[pointer + s] = f"B-{label_name}"
                else:
                    label_list[pointer + s] = f"I-{label_name}"
                pointer += s
                tmp_token = tmp_token[s:]
            except:
                # print(tmp_token)
                print(ttag)
                pass

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
            if ttag.lower() in words_trash:
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


def getHTGen6kRaw():
    return load(
        "results/HTGen_chatgpt_6k.csv",
        ["description"],
        ["gpt_location", "gpt_name"],
        "NAME",
        False,
    )


def getHTGen12kRaw():
    return load(
        "results/HTGen_chatgpt_12k.csv",
        ["description"],
        ["gpt_location", "gpt_name"],
        "NAME",
        False,
    )


def getHTGenV2Raw():
    return load(
        "results/HTGenV2test.csv",
        ["text"],
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

    # Gen6k
    df = load(
        "results/HTGen_chatgpt_6k.csv",
        ["description"],
        ["gpt_location", "gpt_name"],
        "NAME",
    )

    df.to_csv(pj(cache_path, "HTGen-6k_tokenized.csv"), index=False)

    # Gen12k
    df = load(
        "results/HTGen_chatgpt_12k.csv",
        ["description"],
        ["gpt_location", "gpt_name"],
        "NAME",
    )

    df.to_csv(pj(cache_path, "HTGen-12k_tokenized.csv"), index=False)

    # Gen12k
    df = load(
        "results/HTGen_chatgpt_12k.csv",
        ["description"],
        ["gpt_location", "gpt_name"],
        "NAME",
        False,
    )

    df.to_csv(pj(cache_path, "HTGen-12k_aligned.csv"), index=False)

    # GenV2
    df = load(
        "./results/HTGenV2train.csv",
        ["text"],
        ["gpt_location", "gpt_name"],
        "NAME",
    )
    print(df)
    df.to_csv(pj(cache_path, "HTGenV2train_tokenized.csv"), index=False)

    df = load(
        "./results/HTGenV2test.csv",
        ["text"],
        ["gpt_location", "gpt_name"],
        "NAME",
    )
    print(df)
    df.to_csv(pj(cache_path, "HTGenV2test_tokenized.csv"), index=False)
