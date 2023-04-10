import pandas as pd

label_df = pd.read_csv("./data/HTName.csv")  # [:500]
label_df = label_df[
    [
        "title",
        "description",
        "label",
    ]
]

label_df["text"] = label_df.apply(
    lambda x: str(x["title"]) + ". " + str(x["description"]), axis=1
)

label_df["label"] = label_df["label"].fillna("")
label_df["text"] = label_df["text"].apply(lambda x: x.replace("’", "'"))
label_df["label"] = label_df["label"].apply(lambda x: x.replace("’", "'"))

def process():
    from nltk.tokenize import RegexpTokenizer

    tokenizer = RegexpTokenizer(r"[/|(|)|{|}|$|?|!]|\w+|\$[\d\.]+|\S+")
    # tokenizer.tokenize(s.replace('’',"'"))

    delEmpty = lambda x: [] if all(not y for y in x) else x

    tokens, tags = [], []
    for i, line in label_df.iterrows():
        text, label = line["text"], line["label"]
        token = tokenizer.tokenize(text)
        tokens.append(token)

        tag_text = delEmpty(label.split("|"))
        tag_text = [x.strip() for x in tag_text]

        tag = ["O" for _ in range(len(token))]

        if not tag_text:
            tags.append(tag)
            continue

        pointer = 0
        tmp_token = token[::]
        for tag_single in tag_text:
            for ind, ttag in enumerate(tokenizer.tokenize(tag_single)):
                # print(tmp_token)
                s = tmp_token.index(ttag)
                if ind == 0:
                    tag[pointer + s] = "B-NAME"
                else:
                    tag[pointer + s] = "I-NAME"

                pointer += s
                tmp_token = tmp_token[s:]

        tags.append(tag)

    return tokens, tags


def get_raw_csv():
    return label_df


def get_tokenized_csv():
    tokens, tags = process()
    return pd.DataFrame({"tokens": tokens, "tags": tags})


if __name__ == "__main__":

    get_tokenized_csv().to_csv("./data/HTLocation_tokenized.csv", index=False)
