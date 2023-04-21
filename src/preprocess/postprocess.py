from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r"\w+|\$[\d\.]+|\S+")

def pp4Name(le: List[str]):
    """Posprocessing for name entity recognition.
    """
    
    for l in le:
        if 