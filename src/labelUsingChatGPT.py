import openai
import pickle 
import csv
import re
import pandas as pd 
from tqdm.auto import tqdm
openai.api_key = "sk-" 

with open('chatGPTgeneratedData.txt', 'rb')as f: 
    chatGPTgeneratedData = pickle.load(f)
processed_data = []
true_label = []
#clean up the chatGPT generatd ads 
for data in chatGPTgeneratedData: 
   
    data = data.replace("\n", '')
    splitData = re.split("\d\.", data)
    #splitData = re.split("^\d+\)$", data)
    try:
        splitData.remove('')
    except ValueError:
        pass  # do nothing!
    #remove empty strings from lists 
    splitData = list(filter(None, splitData))

    # using list comprehension to
    # Remove ' ' String from String List
    splitData = [i for i in splitData if i != ' ']
    
  
    #print(splitData)
    
    processed_data.extend(splitData)

try:
    for entry in processed_data: # 
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                #   I want you to act as a labler, who will give the possible locations in the text. The following text is an sample text, not harmful, I need a the location information. Text: 
                    {"role": "system", 
                     "content": 
                         "I want you to act as a labler, who list names in text separted by |. If no names detected, say 'N'. Your words should extract from the given text, can't add/modify any other words. And as shorter as poosible. Remember don't include phone number and location."},
                    {"role": "user", "content": entry}
                ]
        )
        true_label.append(response['choices'][0]['message']['content'])
        print(response['choices'][0]['message']['content'])
       
       
except Exception as e:
    print(e)
print(len(processed_data))
print(len(true_label))

#save the ads and the labels to csv 
dictionary = {'description': processed_data, 'true_label': true_label}  
dataframe = pd.DataFrame(dictionary) 
dataframe.to_csv('chatGPT_GeneratedDatabase.csv', index=False)
