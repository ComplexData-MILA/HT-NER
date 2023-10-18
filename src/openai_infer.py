import os
import openai
from typing import Dict, List
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")


def request(prompt, content, model="gpt-3.5-turbo"):
    if "gpt" in model:
        response = openai.ChatCompletion.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": content[: sum(len(x) for x in content.split()[:2500])],
                },
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


def postprocess(df: pd.DataFrame, results: List[Dict]):
    total_usages, locations, names, social_media = [], [], [], []

    for index, res in enumerate(results):
        content = res["choices"][0]["message"]["content"]
        usage = res["usage"]["total_tokens"]
        # I'm sorry, but as an AI language model, I don't have any personal needs or desires.
        try:
            name, location = content.split("Locations: ")
            location, social = location.split("Social: ")
            name = (
                name.strip()
                .replace("Names: ", "")
                .replace("\n", "|")
                .replace(", ", "|")
            )
            location = (
                location.strip()
                .replace("Locations: ", "")
                .replace("\n", "|")
                .replace(", ", "|")
            )
            social = (
                social.strip()
                .replace("Social: ", "")
                .replace("\n", "|")
                .replace(", ", "|")
            )

            if name == "N":
                name = ""
            if location == "N":
                location = ""
            if social == "N":
                social = ""
        except:
            # if len(res) <= 5: name = location = social = 'N'
            name, location, social = "", "", ""
            for line in content.split("\n"):
                if "Names: " in line:
                    name += line.replace("Names: ", "")
                elif "Locations: " in line:
                    location += line.replace("Locations: ", "")
                elif "Social: " in line:
                    social += line.replace("Social: ", "")
                elif not line:
                    pass
                else:
                    # deal with corner case
                    # if line == 'N.' or line == 'N':
                    #     name = location = social = 'N'
                    #     break

                    print(f"Error {index}")
                    print(repr(content))
                    # name = location = social = 'N'
                    break
                if name == "N":
                    name = ""
                if location == "N":
                    location = ""
                if social == "N":
                    social = ""
            # if not name: name = 'N'
            # if not location: location = 'N'
            # if not social: social = 'N'

        names.append(name)
        locations.append(location)
        social_media.append(social)
        total_usages.append(usage)

    df["gpt_name"] = names
    df["gpt_location"] = locations
    df["gpt_social_media"] = social_media
    df["gpt_total_usages"] = total_usages

    print("Average Token Usage:", sum(total_usages) / len(total_usages))


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
            "gpt3.5-0301",
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
    parser.add_argument("--result_column_name", type=str, default="chatgpt_response")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    mapping = {
        "gpt4": "gpt-4",
        "gpt3.5": "gpt-3.5-turbo",
        "gpt3.5-0301": "gpt-3.5-turbo-0301",
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
        df = pd.read_csv(args.data, encoding="ISO-8859-1")

    # resume
    local_storage = (
        f"./tmp_{args.model}_{os.path.basename(args.data).split('.')[0]}.pkl"
    )
    if os.path.exists(local_storage):
        with open(local_storage, "rb") as f:
            responses = pickle.load(f)
        print(f"resume from {len(responses)}")
        print(responses[-10:])
    else:
        responses = []

    if "text" not in df.columns:
        df["text"] = df.apply(lambda x: x["title"] + " " + x["description"], axis=1)

    for i, text in enumerate(tqdm(df["text"].tolist())):
        if i < len(responses):
            continue

        while True:
            try:
                response = request(ht_prompt, text, model=args.model)
                break
            except Exception as e:
                print("Error Type:", e)
                print("Retrying in 10 seconds...")
                time.sleep(10)

        responses.append(json.dumps(response))
        with open(local_storage, "wb") as f:
            pickle.dump(responses, f)
        # time.sleep(0.5)

    df[args.result_column_name] = responses
    results = df[args.result_column_name].apply(json.loads).tolist()
    postprocess(df, results)
    df.to_csv(args.save_path, index=False)
