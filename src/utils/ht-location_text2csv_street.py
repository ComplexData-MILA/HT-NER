import pandas as pd

label_df = pd.read_csv("./data/HTLocation.csv")  # [:500]
label_df = label_df[
    [
        "ad_id",
        "title",
        "description",
        "location",
        "label (PN) different entities sep by |",
        "Label for title",
        "Street Level",
        "City Level",
        "Title Street or Other",
        "Street or Other",
    ]
]
label_df = label_df.rename(
    columns={
        "label (PN) different entities sep by |": "label",
        "Label for title": "title label",
        "Title Street or Other": "title street mask",
        "Street or Other": "street mask",
    }
)

label_df["text"] = label_df.apply(
    lambda x: str(x["title"]) + ". " + str(x["description"]), axis=1
)

label_df["label"] = label_df["label"].fillna("")
label_df["title label"] = label_df["title label"].fillna("")

label_df["Street Level"] = label_df["Street Level"].fillna(0)
label_df["City Level"] = label_df["City Level"].fillna(0)

label_df["text"] = label_df["text"].apply(lambda x: x.replace("’", "'"))
label_df["label"] = label_df["label"].apply(lambda x: x.replace("’", "'"))
label_df["title label"] = label_df["title label"].apply(lambda x: x.replace("’", "'"))
label_df["street mask"] = label_df["street mask"].fillna("").astype(str)
label_df["title street mask"] = label_df["title street mask"].fillna("").astype(str)


def process():
    from nltk.tokenize import RegexpTokenizer

    # s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks. (519) 1293 123"
    tokenizer = RegexpTokenizer(r"[/|(|)|{|}|$|?|!]|\w+|\$[\d\.]+|\S+")
    # tokenizer.tokenize(s.replace('’',"'"))

    delEmpty = lambda x: [] if all(not y for y in x) else x

    tokens, tags = [], []
    for i, line in label_df.iterrows():
        text, label, title_label = line["text"], line["label"], line["title label"]
        title_mask, mask = line["title street mask"], line["street mask"]
        token = tokenizer.tokenize(text)
        tokens.append(token)

        tag_text = delEmpty(title_label.split("|")) + delEmpty(label.split("|"))
        tag_mask = delEmpty(title_mask.split("|")) + delEmpty(mask.split("|"))
        tag_text = [x.strip() for x in tag_text]
        tag = ["O" for _ in range(len(token))]

        if not tag_text or not tag_mask or tag_mask.count("0") == len(tag_mask):
            tags.append(tag)
            continue

        if len(tag_text) != len(tag_mask):
            print(i, tag_text, tag_mask)
        assert len(tag_text) == len(tag_mask)

        pointer = 0
        tmp_token = token[::]
        for tag_single, mask_single in zip(tag_text, tag_mask):
            if mask_single == "0":
                continue
            for ind, ttag in enumerate(tokenizer.tokenize(tag_single)):
                # print(tmp_token)
                s = tmp_token.index(ttag)
                if ind == 0:
                    tag[pointer + s] = "B-LOC"
                else:
                    tag[pointer + s] = "I-LOC"

                pointer += s
                tmp_token = tmp_token[s:]

        tags.append(tag)
        # gaps = 12
        # for i in range(0, len(token), gaps):
        #     if 'B-LOC' in tag[i:i+gaps] or 'I-LOC' in tag[i:i+gaps]:
        #         print('\t'.join(token[i:i+gaps]))
        #         print('\t'.join(tag[i:i+gaps]))

    return tokens, tags


def get_raw_csv():
    return label_df


def get_tokenized_csv():
    tokens, tags = process()
    return pd.DataFrame({"tokens": tokens, "tags": tags})


if __name__ == "__main__":

    get_tokenized_csv().to_csv("./data/HTLocation_tokenized_street.csv", index=False)
