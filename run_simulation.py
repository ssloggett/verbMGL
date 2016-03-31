import corpus_functions
from corpus_functions import *

myTrees = BracketParseCorpusReader(root = 'childes', fileids = '.*\.parsed').parsed_sents()

sentences = build_corpus(myTrees, flat_structure = False)

similarity = get_pos_similarity(myTrees)

simulate(sentences, similarity_matrix = similarity, start =50, max = 300, by = 50, rep = 4, conf=[0], printR = True, test = 100)