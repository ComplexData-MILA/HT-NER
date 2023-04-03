import pandas as pd
label_df = pd.read_csv('./data/location_eval_500ads.csv')[:500]
label_df = label_df[['ad_id', 'title', 'description', 'location', \
    'label (PN) different entities sep by |', 'Label for title','Street Level', 'City Level']]
label_df = label_df.rename(columns={'label (PN) different entities sep by |':'label', 'Label for title': 'title label'})

label_df['text'] = label_df.apply(lambda x: str(x['title']) + ". " + str(x['description']), axis=1)

label_df['label'] = label_df['label'].fillna("")
label_df['title label'] = label_df['title label'].fillna("")

label_df['Street Level'] = label_df['Street Level'].fillna(0)
label_df['City Level'] = label_df['City Level'].fillna(0)

label_df['text'] = label_df['text'].apply(lambda x: x.replace('’',"'"))
label_df['label'] = label_df['label'].apply(lambda x: x.replace('’',"'"))
label_df['title label'] = label_df['title label'].apply(lambda x: x.replace('’',"'"))
        
# print(label_df)

def get_csv():
    return label_df

if __name__ == "__main__":
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r"[/|(|)|{|}|$|?|!]|\w+|\$[\d\.]+|\S+")
    # tokenizer.tokenize(s.replace('’',"'"))

    delEmpty = lambda x: [] if all(not y for y in x) else x

    tokens, tags = [], []
    for i, line in label_df.iterrows():
        text, label, title_label = line['text'], line['label'], line['title label']
        token = tokenizer.tokenize(text)
        tokens.append(token)
        
        tag_text = delEmpty(title_label.split('|')) + delEmpty(label.split('|'))
        tag_text = [x.strip() for x in tag_text]

        tag = ['O' for _ in range(len(token))]
        
        if not tag_text:
            tags.append(tag)
            continue
        
        pointer = 0
        tmp_token = token[::]
        for tag_single in tag_text:
            for ind, ttag in enumerate(tokenizer.tokenize(tag_single)):
                # print(tmp_token)
                s = tmp_token.index(ttag)
                if ind == 0: tag[pointer + s] = 'B-LOC'
                else: tag[pointer + s] = 'I-LOC'
                
                pointer += s
                tmp_token = tmp_token[s:]
        
        tags.append(tag)
        # gaps = 12
        # for i in range(0, len(token), gaps):
        #     if 'B-LOC' in tag[i:i+gaps] or 'I-LOC' in tag[i:i+gaps]:
        #         print('\t'.join(token[i:i+gaps]))
        #         print('\t'.join(tag[i:i+gaps]))
        
    # tokens, tags

    pd.DataFrame({'tokens': tokens, 'tags': tags}).to_csv("./data/ht_tokenized.csv", index=False)