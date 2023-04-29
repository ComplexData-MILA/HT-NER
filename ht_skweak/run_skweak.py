# Imports

import argparse
import pandas as pd
import spacy 
import nltk
nltk.download('punkt')

from collections import Counter
from tqdm import tqdm

from labeling_functions import RuleAnnotator, DictionaryAnnotator, NameDisambiguationAnnotator, FrequencyDetector
from labeling_functions import AllCapsDetector, NameCaseStructureDetector, SpacyAntiNameDetector, NERModelResultDetector, combine_lfs

from neat_preprocess import preprocess
from process_doc import create_doc, get_docs, store_doc_list

from skweak.generative import HMM

data_path = "src/HTName.csv"

namelist_path = "src/nameslist.csv"
dictionary_path = "src/weights.json"
expanded_dictionary_path = "/src/generatedFemaleNamesPlusOriginalDict30000 (1).json"

model_path = "ht_bert_v3"

def main(args):

    # 1. Load and process dataset

    # load spacy model
    nlp = spacy.load("en_core_web_sm")

    # read csv dataset
    text_df = pd.read_csv(args.dataset_path)
    titles = list(text_df['title'])
    descs = list(text_df['description'])
    text = []
    for i in range(len(titles)):
        text.append(titles[i] + " - " + descs[i])
    text = list(map(preprocess, text))
    # create docs
    data_docs = create_doc(text, nlp)

    # 2. Create LFs

    # create lf
    lfs = []

    # add lf you want to run into lfs list above. See example below
    # 0 - don't use, 1 - use, 2 - threshold (if applicable)
    rule = 1
    dictionary = 1
    exp_dictionary = 0
    disambiguation = 0
    cap_disambiguation = 0
    frequency = 1
    all_caps = 0
    name_structure = 1
    spacy_anti = 1

    if rule == 1:
        for i in range(27):
            lfs.append(RuleAnnotator(i))    

    if dictionary == 1:
        lfs = lfs + [DictionaryAnnotator(dictionary_path, "full_dictionary")]
    elif dictionary == 2:
        thresholds = ["q1", "q2", "q3", "q4"]
        for threshold in thresholds:
            lfs.append(DictionaryAnnotator(dictionary_path, threshold + "_thr_dictionary"))

    if exp_dictionary == 1:
        lfs = lfs + [DictionaryAnnotator(dictionary_path, "full_expanded_dictionary")]
    elif exp_dictionary == 2:
        thresholds = ["q1", "q2", "q3", "q4"]
        for threshold in thresholds:
            lfs.append(DictionaryAnnotator(dictionary_path, threshold + "_thr_expanded_dictionary"))

    if disambiguation == 1:
        lfs = lfs + [NameDisambiguationAnnotator(thr = 0.1, add_bound = 0.05, upper_bound = False, weights_dict_path = dictionary_path, model_path = model_path, namelist_path = namelist_path)]
    elif disambiguation == 2:
        thresholds = [0.1, 0.2, 0.3, 0.4]
        for threshold in thresholds:
            lfs.append(NameDisambiguationAnnotator(thr = threshold, add_bound = 0.05, upper_bound = False, weights_dict_path = dictionary_path, model_path = model_path, namelist_path = namelist_path))

    # Extra Rule LFs

    if frequency > 1:
    # all tokens that arent stop words or punctuations
        words = []
        for doc in tqdm(data_docs):  # for doc in data we want to fit hmm on
            words = words + [token.text for token in doc if not token.is_stop and not token.is_punct]
        # get most common tokens
        word_freq = Counter(words)
        sorted_word_freq = [x[0] for x in word_freq.most_common(len(word_freq))]

        # threshold frequency detector
        if frequency == 1:
            lfs.append(FrequencyDetector(sorted_word_freq, 0.01))
        elif frequency == 2:
            thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]
            for threshold in thresholds:
                lfs.append(FrequencyDetector(sorted_word_freq, threshold))
        
    # capital detectors
    if all_caps == 1:
        lfs.append(AllCapsDetector())

    if name_structure == 1:
        lfs.append(NameCaseStructureDetector())

    # antirules
    if spacy_anti == 1:
        lfs.append(SpacyAntiNameDetector(nlp))

    # combine annotators
    combined_annotator = combine_lfs(lfs)

    # 3. Apply LFs
    for doc in tqdm(data_docs):
        combined_annotator(doc)       

    # 4. Aggregate
    # if you want to start with custom starting LF weights, 
    # give a dictionary of initial weights of all LFs used to annotate as argument "initial_weights = dictionary_name"
    # you can get names of all LFs by running: list(annotated_docs[0].spans.keys())
    unified_model = HMM("hmm", labels = ["PERSON_NAME", "NOT_NAME"])
    unified_model.fit(data_docs)

    unified_docs = []
    for doc in tqdm(data_docs):
        unified_docs.append(unified_model(doc))

    for doc in unified_docs:
        doc.ents = doc.spans["hmm"]     

    store_doc_list(unified_docs, "src/SampleAnnotated.spacy")

    # 5. Spacy docs ->  "Results" column in a df
    hmm_preds = []  # list of results
    for doc in unified_docs:
        entities = ''
        for ent in doc.ents:
            if ent.label_ == "PERSON_NAME":
                entities+=ent.text+"|"
        if entities != '':
            hmm_preds.append(entities)
        else:
            hmm_preds.append('N')   

    print("HMM PREDS:", hmm_preds)

    res_df = pd.DataFrame({args.experiment_name + " results": hmm_preds})
    res_df.to_csv(args.results_path, index=False)

if __name__ == "__main__":
    ### Receive Augmentation
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--results_path", type=str)
    parser.add_argument("--experiment_name", type=str)
    args = parser.parse_args()
    main(args)