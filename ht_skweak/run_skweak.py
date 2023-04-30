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
annotated_doc_output_path = "src/annotated_doc.spacy"

def main(args):

    # 1. Load and process dataset

    # load spacy model
    nlp = spacy.load("en_core_web_sm")

    # read csv dataset
    text_df = pd.read_csv(args.dataset_path)
    
    # grab title and description column and change into list of text
    titles = list(text_df['title'])
    descs = list(text_df['description'])

    # combine title and description into a single string
    text = []
    for i in range(len(titles)):
        text.append(titles[i] + " - " + descs[i])

    # preprocess text using NEAT's preprocess code
    text = list(map(preprocess, text))

    # create docs
    data_docs = create_doc(text, nlp)

    # 2. Create LFs

    # create lf
    lfs = []

    # add lf you want to run into lfs list above. All available LFs are in labeling_functions.py 
    # Below is an example of how to add them:

    # 0 - don't use, 1 - use, 2 - threshold (if applicable)
    # NEAT Rules
    rule = 0       
    # NEAT Weighted/Unweighted dictionary, depending on dictionary given in path             
    dictionary = 0         
    # Same as above, but using expanded_dictionary_path   
    exp_dictionary = 0     
    # NEAT Name Extractor using dictionary specified in dictionary_path    
    disambiguation = 1      
    # Antirule: Most frequent words at a given threshold is set as "NOT_NAME"    
    frequency = 0  
    # All caps word is set "PERSON_NAME"             
    all_caps = 0       
    # Words in the middle of a sentence that have capital first letter are set as "PERSON_NAME"         
    name_structure = 0        
    # All spacy model entities that are not labeled "PERSON" is set as "NOT_NAME"
    spacy_anti = 0

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

    # doc.spans is a dictionary where key = LF name, value = LF results.
    # keep the aggregated results of hmm only.
    for doc in unified_docs:
        doc.ents = doc.spans["hmm"]     

    # Store annotated (LF labeled) spacy doc at given path
    store_doc_list(unified_docs, annotated_doc_output_path)

    # 5. Spacy docs ->  "Results" column in a df
    hmm_preds = []  # list of results
    for doc in unified_docs:
        entities = ''
        for ent in doc.ents:
            # Antirules produce "NOT_NAME" labels. Grab "PERSON_NAME" entities only
            if ent.label_ == "PERSON_NAME":
                entities+=ent.text+"|"
        if entities != '':
            # Found "PERSON_NAME" entities, append
            hmm_preds.append(entities)
        else:
            # No "PERSON_NAME" entities, represent using 'N'
            hmm_preds.append('N')   

    res_df = pd.DataFrame({args.experiment_name + " results": hmm_preds})
    res_df.to_csv(args.results_path, index=False)

if __name__ == "__main__":
    ### Receive Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--results_path", type=str)
    parser.add_argument("--experiment_name", type=str)
    args = parser.parse_args()
    main(args)