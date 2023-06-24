import spacy
import skweak
from bs4 import BeautifulSoup
from neat_preprocess import preprocess

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("emoji", first=True)

# 1. Preprocessing

# 1.0 Helpers

def removeHtmlTags(string):
    soup = BeautifulSoup(string, 'html.parser')
    return soup.get_text()

# 1.1. Creating, storing, retrieving Doc objects

def create_doc(docs, spacy_model): # list of strings --> doc object
    # remove HTML Tags and clean using NEAT preprocess func
    html_free_docs = list(map(removeHtmlTags, map(preprocess,docs)))
    return list(spacy_model.pipe(html_free_docs))

def store_doc_list(docs, path): # eg path "path/to/your/file.spacy"
    # store a list of Doc documents into a single file 
    skweak.utils.docbin_writer(docs, path)

def get_docs(path):
    # retrieve this list
    return list(skweak.utils.docbin_reader(path))