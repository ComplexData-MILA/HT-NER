import spacy
import json
import pandas as pd
import re
from spacy.matcher import Matcher, PhraseMatcher
from neat_base_classes import Extractor, Entity
from neat_disambiguation import FillMaskFilter
from neat_preprocess import preprocess


class RuleExtractor(Extractor):
    def __init__(self, pnum, **kwargs):
        # pnum == -1 -> define_all_patterns
        model = kwargs.pop('model', 'en_core_web_sm')
        Extractor.__init__(self, model)
        self.patterns, self.weights = self.define_patterns(pnum)
        self.matcher = self.create_matcher()
        self.type = 'rule'
        
    def create_matcher(self):
        matcher = Matcher(self.nlp.vocab)
        for k,v in self.patterns.items():
            matcher.add(k,[v])
        return matcher
    
    def define_patterns(self, pnum):
        # English patterns
        pattern0 = [{"LOWER": "call"}, {"LOWER": "me"},{"TAG": "NNP"}]
        pattern1 = [{"LOWER": "name"}, {"LOWER": "is"},{"TAG": "NNP"}]
        pattern2 = [{"LOWER": "i"}, {"LOWER":"am"},{"TAG": "NNP"}]
        pattern3 = [{"LOWER": "it"}, {"LOWER":"is"},{"TAG": "NNP"}]
        pattern4 = [{"LOWER": "ask"}, {"LOWER":"for"},{"TAG": "NNP"}]
        pattern5 = [{"LOWER":"Ms"},{"TAG": "NNP"}]
        pattern6 = [{"LOWER":"ms."},{"TAG": "NNP"}]
        pattern7 = [{"LOWER":"aka"},{"TAG": "NNP"}]
        pattern8 = [{"LOWER":"miss"},{"TAG": "NNP"}]
        pattern9 = [{"LOWER":"Miss."},{"TAG": "NNP"}]
        pattern10 = [{"LOWER":"Ts"},{"TAG": "NNP"}]
        pattern11 = [{"LOWER":"Mrs"},{"TAG": "NNP"}]
        pattern12 = [{"LOWER":"mrs."},{"TAG": "NNP"}]
        pattern13 = [{"LOWER":"Mz"},{"TAG": "NNP"}]
        pattern14 = [{"LOWER":"mz."},{"TAG": "NNP"}]
        pattern15 = [{"LOWER":"named"},{"TAG": "NNP"}]

        # French patterns
        pattern16 = [{"LOWER": "appelez"}, {"LOWER": "moi"},{"TAG": "NNP"}]  # call me NNP
        pattern17 = [{"LOWER": "appelle"}, {"LOWER": "moi"},{"TAG": "NNP"}] # call me NNP
        pattern18 = [{"LOWER": "nom"}, {"LOWER": "est"},{"TAG": "NNP"}] # name is NNP
        pattern19 = [{"LOWER": "m"}, {"LOWER": "appelle"},{"TAG": "NNP"}]   # name is NNP
        pattern20 = [{"LOWER": "c"}, {"LOWER": "est"},{"TAG": "NNP"}]   # it is NNP
        pattern21 = [{"LOWER":"demander"},{"TAG": "NNP"}]   # ask for NNP
        pattern22 = [{"LOWER":"Mme"},{"TAG": "NNP"}]    # Ms NNP
        pattern23 = [{"LOWER":"Madame"},{"TAG": "NNP"}] # Ms NNP
        pattern24 = [{"LOWER":"Mademoiselle"},{"TAG": "NNP"}]   # Miss NNP
        pattern25 = [{"LOWER":"alias"},{"TAG": "NNP"}]  # aka NNP
        pattern26 = [{"LOWER":"surnom"},{"TAG": "NNP"}] # aka NNP
    
        patterns={'pattern0':pattern0, 'pattern1':pattern1,'pattern2':pattern2,
               'pattern3':pattern3,'pattern4':pattern4,'pattern5':pattern5,
               'pattern6':pattern6,'pattern7':pattern7,'pattern8':pattern8,
               'pattern9':pattern9,'pattern10':pattern10,'pattern11':pattern11,
               'pattern12':pattern12,'pattern13':pattern13,'pattern14':pattern14,
                'pattern15':pattern15,'pattern16':pattern16,'pattern17':pattern17,
                'pattern18':pattern18,'pattern19':pattern19,'pattern20':pattern20,
                'pattern21':pattern21,'pattern22':pattern22,'pattern23':pattern23,
                'pattern24':pattern24,'pattern25':pattern25,'pattern26':pattern26,}
        weights = {'pattern0':(2, 0.5), 'pattern1':(2, 0.67),'pattern2':(2, 0.44),
               'pattern3':(2, 0.35),'pattern4':(2, 0.72),'pattern5':(1, 0.5),
               'pattern6':(1, 0.5),'pattern7':(1, 0.5),'pattern8':(1, 0.67),
               'pattern9':(1, 0.5),'pattern10':(1, 0.5),'pattern11':(1, 0.5),
               'pattern12':(1, 0.5),'pattern13':(1, 0.5),'pattern14':(1, 0.5),
                'pattern15':(1, 0.75),'pattern16':(2, 0.5),'pattern17':(2, 0.5),
                'pattern18':(2, 0.67),'pattern19':(2, 0.67),'pattern20':(2, 0.35),
                'pattern21':(1, 0.72),'pattern22':(1, 0.5),'pattern23':(1, 0.5),
                'pattern24':(1, 0.5),'pattern25':(1, 0.5),'pattern26':(1, 0.5),}
        if pnum == -1:
          return patterns, weights
        else:
          p = {"pattern"+str(pnum): patterns["pattern"+str(pnum)]}
          w = {"pattern"+str(pnum): weights["pattern"+str(pnum)]}
          return (p, w)
        
    def extract(self, text):
        if type(text)==float:
            return []
        doc = self.nlp(text)
        matches = self.matcher(doc)
        result = []
        for match_id, start, end in matches:
            string_id = self.nlp.vocab.strings[match_id]
            name_start=start+self.weights[string_id][0]
            span = doc[name_start:end] 
            ent = Entity(span.text,span.start, self.type)
            ent.base_conf = self.weights[string_id][1]
            ent.confidence = ent.base_conf
            ent.type = 'rule'
            result.append(ent)
        return result
    
    def skweak_extract(self, doc):
        matches = self.matcher(doc)
        result = []
        for match_id, start, end in matches:
            string_id = self.nlp.vocab.strings[match_id]
            name_start=start+self.weights[string_id][0]
            span = doc[name_start:end] 
            ent = Entity(span.text,span.start, self.type)
            ent.base_conf = self.weights[string_id][1]
            ent.confidence = ent.base_conf
            ent.type = 'rule'
            result.append(ent)
        return result

class DictionaryExtractor(Extractor):
    def __init__(self,**kwargs):
        model = kwargs.pop('model', 'en_core_web_sm')
        Extractor.__init__(self, model)
        """try:
            # load dictionary with weights if possible
            weights_file = kwargs.pop('weights_dict', dictionary_path)
            self.weights = self.load_weight_dict(weights_file)
            self.terms = list(self.weights.keys())
        except:
            # else load the default dictionary and set every word's weight as 0.5
            try:
                self.terms = kwargs.pop('dictionary')
            except:
                dict_file = kwargs.pop('dict_file', namelist_path)
                self.terms = self.load_word_dict(dict_file)
            self.weights = {n:0.5 for n in self.terms}
        """
        weights_file = kwargs.pop('weights_dict_path')
        self.weights = self.load_weight_dict(weights_file)
        self.terms = list(self.weights.keys())
        self.matcher = self.create_matcher()
        self.type = 'dict'

    def load_weight_dict(self, filename):
        with open(filename) as json_file:
            weights = json.load(json_file)
        return weights

    def load_word_dict(self,dict_file): 
        df_names = pd.read_csv(dict_file)
        df_names.drop_duplicates(subset='Name', inplace=True)
        df_names.dropna()
        newwordlist = df_names['Name']
        return list(set([word.strip().lower() for word in newwordlist]))
    
    def create_matcher(self):
        matcher = PhraseMatcher(self.nlp.vocab,attr="LOWER")
        if len(self.terms)>=len(self.weights):
            patterns = [self.nlp.make_doc(text) for text in self.terms]
        else:
            patterns = [self.nlp.make_doc(text) for text in self.weights.keys()]
        matcher.add('namelist', None, *patterns)
        return matcher
        
    def extract(self, text):
        doc = self.nlp(text)
        matches = self.matcher(doc)
        result = []
        for _, start, end in matches:
            span = doc[start:end]
            ent = Entity(span.text,span.start, self.type)
            ent.base_conf = self.weights[ent.text.lower()]
            ent.confidence = ent.base_conf
            ent.type = 'dict'
            result.append(ent)
        return result

    def skweak_extract(self, doc):
        matches = self.matcher(doc)
        result = []
        for _, start, end in matches:
            span = doc[start:end]
            ent = Entity(span.text,span.start, self.type)
            ent.confidence = ent.base_conf
            ent.type = 'dict'
            result.append(ent)
        return result
    
class NameExtractor(Extractor):
    def __init__(self, threshold=0.10, add_bound=0.05, upper_bound = False, **kwargs):
        """
        Initialize the dictionary and rule extractors
        Args:
            threshold: a float that controls the confidence score used in filtering out the output.
        Returns:
        """
        self.dict_extractor = DictionaryExtractor(**kwargs)
        self.rule_extractor = RuleExtractor(-1, **kwargs)
        self.fillMaskFilter = FillMaskFilter(**kwargs)
        self.threshold = threshold
        self.add_bound = add_bound
        self.upper_bound = upper_bound

    def find_ent(self, target_word, ent_list):
        """
        Return the entity if it is the same as the target entity. Inputs should guarantee there will be a match.s
        Args:
            target_word: A string value that holds the word you want to search.
            ent_list: A set of Entities that you want to search from.
        Returns:
        """
        for e in ent_list:
            if target_word==e:
                return e
        return None

    def compute_combined(self, dict_res, rule_res):
        """
        Compute the confidence score for each predicted word from the base extractors.
        Args:
            dict_res: A set of Entities extracted from the dictionary extractor.
            rule_res: A set of Entities extracted from the rule extractor.
        Returns:
            A list that contians all unique Entities with the combined confidence from the base extractors.
        """
        intersection = dict_res & rule_res
        unilateral = (dict_res - rule_res) | (rule_res - dict_res)

        for res in intersection:
            res.base_conf = self.find_ent(res, dict_res).base_conf*0.5 + self.find_ent(res, rule_res).base_conf*0.5 
        for res in unilateral:
            res.base_conf = self.find_ent(res, unilateral).base_conf*0.5
                
        total_res = list(intersection | unilateral)
        
        return total_res

    def extract(self, text, preprocess_text=True):
        """
            Extracts information from a text using NEAT.
        Args:
            text (str): the text to extract from. Usually a piece of ad description or its title.
            preprocess(bool): set to True if the input text needs preprocessing before the extraction. Default is True.
        Returns:
            List(Entity): a list of entities or the empty list if there are no extracted names.
        """
        if preprocess_text:
            text = preprocess(text)
        dict_res = set(self.dict_extractor.extract(text))
        rule_res = set(self.rule_extractor.extract(text))
        results = self.compute_combined(dict_res, rule_res)
        
        # pass to the disambiguation layer        
        results_text = [result.text for result in results]
        text = re.sub(r'[\.,]+',' ',text)
        filtered_results = self.fillMaskFilter.disambiguate_layer(text, results_text)
      

        # add the disambiguated ratio
        conf_dict = {} # key: entity   value: [confidence, fill_mask_conf, context]
        for result, filtered in zip(results, filtered_results):
            if result not in conf_dict:
                conf_dict[result] = [result.base_conf, filtered['ratio'], [filtered['context']]]
            else:
                conf_dict[result][0]  *= result.base_conf
                conf_dict[result][1]  *= filtered['ratio']
                conf_dict[result][2].append(filtered['context'])

        entity_list = []
        # compute and record the confidence score in the "confidence" field
        for ent, conf_list in conf_dict.items():
            ent.base_conf = conf_list[0]
            ent.fill_mask_conf = conf_list[1]
            ent.context = conf_list[2]
            ent.confidence = ent.base_conf*0.5+ent.fill_mask_conf*0.5
            if self.upper_bound:
              if ent.confidence >= self.threshold and ent.confidence < self.threshold + self.add_bound:
                entity_list.append(ent)
            else:
              if ent.confidence >= self.threshold:
                entity_list.append(ent)

        return entity_list
