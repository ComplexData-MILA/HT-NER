from typing import Dict, List
import pandas as pd

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
            
            if name == "N": name = ""
            if location == "N": location = ""
            if social == "N": social = ""
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
                if name == "N": name = ""
                if location == "N": location = ""
                if social == "N": social = ""
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

    print(sum(total_usages) / len(total_usages))


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--result_column_name", type=str, default="chatgpt_response")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    results = df[args.result_column_name].apply(json.loads).tolist()
    postprocess(df, results)
    df.to_csv(args.save_path, index=False)
