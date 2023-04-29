import spacy

# Original NEAT extractor class
class Extractor(object):
    """
    All extractors extend this abstract class.
    """

    def __init__(self, model='en_core_web_sm'):
        self.nlp = spacy.load(model)

    def extract(self, *input_value, **configs):
        """
        Args:
            input_value (): some extractors may want multiple arguments, for example, to
            concatenate them together
        Returns: list of extracted data as String. Returns an empty list if extractors fail
            to extract any data.
        """
        pass

# Original NEAT entity class
class Entity:
    def __init__(self, text: str, begin_offset: int, type):
        """Construct an entity object.
        text: the plain text of this entity.
        score: The certainty of this entity being a name. 
            Default is -1.0 meaning the certainty is not calculated by word embedding.
        begin_offset: The index of the entity within the parent document.
        """
        self.text = text
        # self.score = score
        self.begin_offset = begin_offset
        self.end_offset = begin_offset + len(text)
        self.type = type
        self.fill_mask_conf = 0
        self.base_conf = 0
        self.confidence = 0
        self.context = []
        

    def __len__(self):
        """The number of unicode characters in the entity, i.e. `entity.text`.
        RETURNS (int): The number of unicode characters in the entity.
        """
        return self.text.length

    def __eq__(self, other):
        """Two entities will be equal to each other if they have the same text and 
        same begin offset. 
        """
        return self.text == other.text and self.begin_offset == other.begin_offset

    def __hash__(self):
        return hash((self.text, self.begin_offset))

    def __str__(self):
        return 'text: '+ self.text + ' confidence: ' + str(self.confidence)