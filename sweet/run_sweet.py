# Imports

import argparse
import pandas as pd
import spacy 
import nltk
nltk.download('punkt')

from tqdm import tqdm
from collections import Counter

from create_lfs import create_sweet_antirules, process_ft_mod_results
from neat_preprocess import preprocess
from process_doc import create_doc, get_docs, store_doc_list
from labeling_functions import FrequencyDetector, NERModelResultDetector, combine_lfs

from skweak.generative import HMM

def main(args):

    # 0. Set input/output paths
    # csv file containing results of individual ft models on the dataset used for fine-tuned model LFs. 
    fine_tuned_model_results_path = args.ft_model_results_path#"src/HTGen_finetune_result.csv"
    # where to save annotated docs. Use 'None' to skip saving.
    annotated_doc_output_path = "src/annotated_doc.spacy"
    # where to save the final results of SWEET
    csv_output_path = args.results_path

    # 1. Load and process dataset

    # load spacy model
    nlp = spacy.load("en_core_web_sm")

    # create docs
    # read csv dataset
    text_df = pd.read_csv(args.dataset_path)
    # grab text that should be processed, convert into list of text
    text = list(text_df['text'])
    # preprocess text using NEAT's preprocess code and convert into spacy docs using loaded nlp model
    text = list(map(preprocess, text))  
    data_docs = create_doc(text, nlp)

    # 2.1 Create Antirules LFs
    sweet_antirules = create_sweet_antirules(data_docs)    

    # 2.2 Create FT Model LFs, and apply all LFs

    # get list of names of ft models used, and a dictionary of its outputs. 
    # The dictionary contains model names as keys, and a processed list of its predictions as values
    ft_mods, ft_mod_outputs = process_ft_mod_results(data_docs, fine_tuned_model_results_path)

    for e, doc in enumerate(tqdm(data_docs)):
        sweet_antirules(doc)
        for ft_mod in ft_mods:
            NERModelResultDetector(ft_mod_outputs[ft_mod][e], ft_mod, False)(doc)

    # 4. Aggregate
    # you can get names of all LFs used by running: list(annotated_docs[0].spans.keys())

    unified_model = HMM("hmm", labels = ["PERSON_NAME", "NOT_NAME"])
    unified_model.fit(data_docs)

    unified_docs = []
    for doc in tqdm(data_docs):
        unified_docs.append(unified_model(doc))

    # doc.spans is a dictionary where key = LF name, value = LF results.
    # keep the aggregated results of hmm only.
    for doc in unified_docs:
        doc.ents = doc.spans["hmm"]     

    # Store annotated (LF labeled) spacy doc at given path unless None
    if annotated_doc_output_path != None:
        store_doc_list(unified_docs, annotated_doc_output_path)

    # 5. Convert annotated spacy docs to "Results" column in a df
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

    res_df = pd.DataFrame({"sweet_results": hmm_preds})

    res_df.to_csv(args.results_path, index=False)
    
if __name__ == "__main__":
    ### Receive Arguments
    parser = argparse.ArgumentParser()
    # path to csv file with a column titled 'text' containing text to run sweet on
    parser.add_argument("--dataset_path", type=str)
    # path to csv file containing predictions of fine-tuned models to be used as LFs
    parser.add_argument("--ft_model_results_path", type=str)
    # path to save csv file containing final results
    parser.add_argument("--results_path", type=str)
    args = parser.parse_args()
    main(args)