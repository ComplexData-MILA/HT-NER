import pandas as pd

label_df = pd.read_csv("./results/HTUnsup_gpt.csv")  # [:500]
label_df = label_df[["title", "description", "gpt_location", "gpt_name"]]

label_df["text"] = label_df.apply(
    lambda x: str(x["title"]) + ". " + str(x["description"]), axis=1
)

label_df["gpt_location"] = label_df["gpt_location"].fillna("")
label_df["gpt_name"] = label_df["gpt_name"].fillna("")
label_df["text"] = label_df["text"].apply(lambda x: x.replace("’", "'"))


def process():
    from nltk.tokenize import RegexpTokenizer

    tokenizer = RegexpTokenizer(r"[/|(|)|{|}|$|?|!]|\w+|\$[\d\.]+|\S+")
    # tokenizer.tokenize(s.replace('’',"'"))

    delEmpty = lambda x: [] if all(not y for y in x) else x

    def f(tag_text, tag, label):
        pointer = 0
        tmp_token = token[::]
        for tag_single in tag_text:
            for ind, ttag in enumerate(tokenizer.tokenize(tag_single)):
                # print(tmp_token)
                s = tmp_token.index(ttag)
                if ind == 0:
                    tag[pointer + s] = f"B-{label}"
                else:
                    tag[pointer + s] = f"I-{label}"

                pointer += s
                tmp_token = tmp_token[s:]
        return tag

    tokens, tags = [], []
    for i, line in label_df.iterrows():
        text, label_loc, label_name = (
            line["text"],
            line["gpt_location"],
            line["gpt_name"],
        )
        token = tokenizer.tokenize(text)
        tokens.append(token)

        # process tag
        tag = ["O" for _ in range(len(token))]

        tag_text_name = [x.strip() for x in delEmpty(label_name.split("|"))]
        if tag_text_name:
            f(tag_text_name, tag, "NAME")

        tag_text_loc = [x.strip() for x in delEmpty(label_loc.split("|"))]
        if tag_text_loc:
            f(tag_text_loc, tag, "LOC")

        tags.append(tag)

    return tokens, tags


def get_raw_csv():
    return label_df


def get_tokenized_csv():
    tokens, tags = process()
    return pd.DataFrame({"tokens": tokens, "tags": tags})


if __name__ == "__main__":

    get_tokenized_csv().to_csv("./data/HTLocation_tokenized.csv", index=False)
