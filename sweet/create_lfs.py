import pandas as pd

from labeling_functions import RuleAnnotator, DictionaryAnnotator, NameDisambiguationAnnotator, FrequencyDetector
from labeling_functions import AllCapsDetector, NameCaseStructureDetector, SpacyAntiNameDetector, NERModelResultDetector
from labeling_functions import SpacyNameDetector, combine_lfs

from neat_preprocess import preprocess

from collections import Counter

namelist_path = "src/nameslist.csv"
dictionary_path = "src/original_dictionary12098.json"
expanded_dictionary_path = "src/expanded_dictionary29234.json"
disambiguation_dictionary_path = dictionary_path
model_path = "ht_bert_v3"

def create_sweet_antirules(data_docs):
    lfs = []

    # 1. Antirule LFs: Most frequent words at a given threshold is set as "NOT_NAME" 
    # get a list of words in the dataset, sorted by most common to least
    words = []
    for doc in data_docs:  # for doc in data we want to fit hmm on
        words = words + [token.text for token in doc if not token.is_stop and not token.is_punct]
    word_freq = Counter(words)
    sorted_word_freq = [x[0] for x in word_freq.most_common(len(word_freq))]

    # create the antirule LFs
    antirules = []
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]
    for threshold in thresholds:
        antirules.append(FrequencyDetector(sorted_word_freq, threshold))

    return combine_lfs(antirules)

def process_ft_mod_results(data_docs, fine_tuned_model_results_path):
    # read df containing ft model results. Multiple lables for a sample is separated by "|", no label indicated by "N"
    ft_results_df = pd.read_csv(fine_tuned_model_results_path).fillna('N')
    # get list of names of ft models used
    ft_mods = list(ft_results_df.columns.values)
    # dictionary with ft model names as key, and a cleaned list of its predicted entities as value
    ft_mod_outputs = {}

    for ftmod in ft_mods:
        preds = list(ft_results_df[ftmod])
        fixed_p = []
        for p in preds:
            if p == 'N':
                fixed_p.append([])
            else:
                new_p = p.replace(" ", "|").split("|")
                new_p = list(map(lambda x: preprocess(x).lower(), new_p))
                fixed_p.append(list(set(new_p)))
        ft_mod_outputs[ftmod] = fixed_p

    return ft_mods, ft_mod_outputs


def create_lfs(data_docs, nlp, rule = 0, dictionary = 0, exp_dictionary = 0, disambiguation = 0, frequency = 0, all_caps = 0, name_structure = 0, spacy_anti = 0, spacy_name = 0):
    lfs = []

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
        lfs = lfs + [DictionaryAnnotator(expanded_dictionary_path, "full_expanded_dictionary")]
    elif exp_dictionary == 2:
        thresholds = ["q1", "q2", "q3", "q4"]
        for threshold in thresholds:
            lfs.append(DictionaryAnnotator(expanded_dictionary_path, threshold + "_thr_expanded_dictionary"))

    if disambiguation == 1:
        lfs = lfs + [NameDisambiguationAnnotator(thr = 0.1, add_bound = 0.05, upper_bound = False, weights_dict_path = disambiguation_dictionary_path, model_path = model_path, namelist_path = namelist_path)]
    elif disambiguation == 2:
        thresholds = [0.1, 0.2, 0.3, 0.4]
        for threshold in thresholds:
            lfs.append(NameDisambiguationAnnotator(thr = threshold, add_bound = 0.05, upper_bound = False, weights_dict_path = disambiguation_dictionary_path, model_path = model_path, namelist_path = namelist_path))

    # Extra Rule LFs

    if frequency > 1:
        words = []
        for doc in data_docs:  # for doc in data we want to fit hmm on
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

    if spacy_name == 1:
        lfs.append(SpacyNameDetector(nlp))

    # antirules
    if spacy_anti == 1:
        lfs.append(SpacyAntiNameDetector(nlp))

    # combine annotators
    combined_annotator = combine_lfs(lfs)

    return combined_annotator

