import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def request(prompt, content, model="gpt-3.5-turbo"):
    if "gpt" in model:
        response = openai.ChatCompletion.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content[:sum(len(x) for x in content.split()[:2500])]},
            ],
            max_tokens=50,
        )
    else:
        response = openai.Completion.create(
            model=model,
            prompt=prompt + "\n" + content,
            temperature=0.0,
            max_tokens=50,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
    return response


if __name__ == "__main__":
    import argparse
    import pandas as pd
    from tqdm import tqdm
    import pickle, json, time
    
    ht_prompt = "I want you to act as a natural and no-bias labler, extract human's name and location or address and social media link or tag in format 'Names: \nLocations: \nSocial: '. If exists multiple entities, spereated by |. If not exists, say N. Your words should extract from the given text, can't add/modify any other words. As shorter as poosible, remember don't include phone number. For one name, should be less than 3 words."

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "gpt4",
            "gpt3.5",
            "davinci",
            "D",
            "Curie",
            "C",
            "Babbage",
            "B",
            "Ada",
            "A",
        ],
        default="gpt3.5",
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--result_column_name", type=str, default="chat_response")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    mapping = {
        "gpt4": "gpt-4",
        "gpt3.5": "gpt-3.5-turbo",
        "davinci": "text-davinci-003",
        "D": "text-davinci-003",
        "Curie": "curie-001",
        "C": "curie-001",
        "Babbage": "babbage-001",
        "B": "babbage-001",
        "Ada": "ada-001",
        "A": "ada-001",
    }
    args.model = mapping[args.model]
    
    try:
        df = pd.read_csv(args.data)
    except Exception as e:
        print(e)
        df = pd.read_csv(args.data, encoding = "ISO-8859-1")
        
    # resume
    local_storage = f"./tmp_{args.model}_{os.path.basename(args.data).split('.')[0]}.pkl"
    if os.path.exists(local_storage):
        with open(local_storage, "rb") as f:
            responses = pickle.load(f)
        print(f"resume from {len(responses)}")
        print(responses[-10:])
    else:
        responses = []
        
    if 'text' not in df.columns:
        df['text'] = df.apply(lambda x: x['title']  + ' ' + x['description'], axis=1)
        
    for i, text in enumerate(tqdm(df["text"].tolist())):
        if i < len(responses):
            continue
        try:
            response = request(ht_prompt, text, model=args.model)
        except Exception as e:
            print(e)
            exit()

        responses.append(json.dumps(response))
        with open(local_storage, "wb") as f:
            pickle.dump(responses, f)
        # time.sleep(0.5)
        
    df[args.result_column_name] = responses
    df.to_csv(args.save_path, index=False)
