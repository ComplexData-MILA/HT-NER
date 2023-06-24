import skweak
from neat_extractors import RuleExtractor, DictionaryExtractor, NameExtractor
# Helpers

def combine_lfs(lfs):
    combined = skweak.base.CombinedAnnotator()
    for lf in lfs:
        combined.add_annotator(lf)
    return combined

def apply_lf(lf, doc, single = True):
    if single:
        return lf(doc)
    else:
        # lf is combined annotator
        return list(lf.pipe(doc))
    
# LFs

# 1.1 ORIGINAL NEAT LFs

class RuleAnnotator(skweak.base.SpanAnnotator):
    def __init__(self, rule_num):
        super(RuleAnnotator, self).__init__("rule"+str(rule_num)+"_annotator")
        self.rule_extractor = RuleExtractor(rule_num)

    def find_spans(self, doc):
        names = [a.text for a in self.rule_extractor.skweak_extract(doc)]
        for tok in doc:
          if tok.text in names:
            yield tok.i, tok.i + 1, "PERSON_NAME"

class DictionaryAnnotator(skweak.base.SpanAnnotator):
    def __init__(self, dictionary_path, threshold):
        super(DictionaryAnnotator, self).__init__(str(threshold)+"_dictionary_locantoscore_annotator")
        self.dict_extractor = DictionaryExtractor(weights_dict_path = dictionary_path)

    def find_spans(self, doc):
        names = [a.text for a in self.dict_extractor.skweak_extract(doc)]
        for tok in doc:
          if tok.text in names:
            yield tok.i, tok.i + 1, "PERSON_NAME"

# original NEAT NameExtractor annotator
class NameDisambiguationAnnotator(skweak.base.SpanAnnotator):
    def __init__(self, thr, add_bound, upper_bound, **kwargs):
        super(NameDisambiguationAnnotator, self).__init__(str(thr)+"_thresh_disambiguation_annotator")
        self.name_extractor = NameExtractor(thr, add_bound, upper_bound, **kwargs)

    def find_spans(self, doc):
        names = [a.text for a in self.name_extractor.extract(doc.text)]
        for tok in doc:
          if tok.text in names:
            yield tok.i, tok.i + 1, "PERSON_NAME"

# 1.2 EXTENDED NEAT LFs

# Multiple run NEAT Fill Mask for capitalized rules disambiguation
class MultCapDisambiguationAnnotator(skweak.base.SpanAnnotator):
    def __init__(self, scored_entities, thr, top_k, mult):
        super(MultCapDisambiguationAnnotator, self).__init__(str(thr)+"thr_top"+str(top_k)+"_"+str(mult)+"mult_cap_disambiguation")
        self.entities =scored_entities
        self.thr = thr

    def find_spans(self, doc):
        elist = self.entities
        elist = elist[elist[:,2]>self.thr]
        for tok in doc:
          if tok.text.lower() in elist[:,1]:
            yield tok.i, tok.i + 1, "PERSON_NAME" 

# 2. Rule based LF

# To do with capitalization
# Word is mid sentence and all caps
class AllCapsDetector(skweak.base.SpanAnnotator):
    def __init__(self):
        super(AllCapsDetector, self).__init__("all_caps_detector")

    def find_spans(self, doc):
        for tok in doc[1:]:
          if tok.is_upper and len(tok.text)>2:
            yield tok.i, tok.i+1, "PERSON_NAME"

# Word is mid sentence and first letter is capitalized
class NameCaseStructureDetector(skweak.base.SpanAnnotator):
    def __init__(self):
        super(NameCaseStructureDetector, self).__init__("name_case_struc_detector")

    def find_spans(self, doc):
        for tok in doc[1:-2]:
          idx = tok.i
          prev = doc[idx - 1].text
          after = doc[idx + 1].text
          cur = tok.text
          if cur[0].isupper() and len(cur)>2 and len(cur)<15:
            yield tok.i, tok.i+1, "PERSON_NAME"

# 3 NER MODEL 
# takes in results of other NER models
class NERModelResultDetector(skweak.base.SpanAnnotator):
    def __init__(self, model_result, model_name, anti = True):
        super(NERModelResultDetector, self).__init__(str(model_name)+"_model_detector")
        self.model_result = model_result
        self.anti = anti

    def find_spans(self, doc):
        if self.model_result == []:
          return
        for tok in doc:
          if tok.text.lower() in self.model_result:
            if self.anti:
              yield tok.i, tok.i+1, "NOT_NAME"
            else:
              yield tok.i, tok.i+1, "PERSON_NAME"

# spacy PERSON entities
class SpacyNameDetector(skweak.base.SpanAnnotator):
    def __init__(self, spacy_model):
        super(SpacyNameDetector, self).__init__("spacy_name_detector")
        self.model = spacy_model

    def find_spans(self, doc):
        names = []
        for e in self.model(doc.text).ents:
          if e.label_ == "PERSON":
            names.append(e.text)
        for tok in doc:
          if tok.text in names:
            yield tok.i, tok.i+1, "PERSON_NAME"

# 4 ANTIRULES

# Words that appear many times in the corpus shouldn't be a name
class FrequencyDetector(skweak.base.SpanAnnotator):
    def __init__(self, sorted_word_freq, threshold):
        super(FrequencyDetector, self).__init__(str(threshold)+"_frequent_word_detector")
        self.common_words = sorted_word_freq[0:int(len(sorted_word_freq)*threshold)]

    def find_spans(self, doc):
        for tok in doc:
          if tok.text in self.common_words:
            yield tok.i, tok.i+1, "NOT_NAME"

# spacy non-PERSON entities
class SpacyAntiNameDetector(skweak.base.SpanAnnotator):
    def __init__(self, spacy_model):
        super(SpacyAntiNameDetector, self).__init__("spacy_antiname_detector")
        self.model = spacy_model

    def find_spans(self, doc):
        antinames = []
        for e in self.model(doc.text).ents:
          if e.label_ != "PERSON":
            antinames.append(e.text)
        for tok in doc:
          if tok.text in antinames:
            yield tok.i, tok.i+1, "NOT_NAME"