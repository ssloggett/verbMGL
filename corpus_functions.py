import nltk
from nltk import *
from nltk.corpus import treebank
import nltk.tree
from nltk.tree import *
import nltk.corpus.reader.bracket_parse
from nltk.corpus.reader.bracket_parse import *

import math
from math import log

import verbMGL
from verbMGL import *

def is_parent(node, subtree):
    if subtree.parent() is not None and subtree.parent().label() == node: return True
    return False

def is_grandparent(node, subtree):
    if subtree.parent().parent() is not None and subtree.parent().parent().label() == node: return True
    return False

def subject_tag(tree):
    tree = ParentedTree.convert(tree)
    subjects = []
    for subtree in [x for x in tree.subtrees()]:
        if subtree.right_sibling() is not None:
            if subtree.label() == 'NP' and subtree.right_sibling().label() == 'VP' and (is_parent('S', subtree) or is_parent('SQ', subtree) or is_grandparent('SQ', subtree)):
                subjects.append(subtree)
    for subject in subjects:
        subject.set_label('NP-SUBJ')
        subj_heads = ['NN', 'NNS', 'PRP', 'NNP', 'NNPS']
        for preterminal in subject.subtrees():
            subj_head = False
            if preterminal.label() in subj_heads:
                if is_parent('NP-SUBJ', preterminal): subj_head = True
                elif is_grandparent('NP-SUBJ', preterminal) and preterminal.right_sibling() is None: subj_head = True
                elif is_grandparent('NP-SUBJ', preterminal) and preterminal.right_sibling() != 'POS': subj_head = True
            if subj_head:
                if preterminal.label() == 'NNP': preterminal.set_label('NN-SUBJ')
                elif preterminal.label() == 'NNPS': preterminal.set_label('NNS-SUBJ')
                else: preterminal.set_label(preterminal.label() + '-SUBJ')
    return Tree.convert(tree)

def convert_tree(tree):
    tree = ParentedTree.convert(tree)
    subtrees = [x for x in tree.subtrees()]
    open_nodes, closed_nodes, new_tree = [], [], []
    for subtree in subtrees:
        sub_subtrees = [x for x in subtree.subtrees()]
        if len(sub_subtrees) > 1:
            open_nodes.insert(0,subtree.treeposition())
            new_tree.append(['[', subtree.label()])
        else:
            new_tree.append([subtree.leaves()[0], subtree.label()])
            closed_nodes.append(subtree.treeposition())
        for node in open_nodes:
            sub_nodes = [x.treeposition() for x in tree[node].subtrees() if x is not tree[node]]
            if close_check(sub_nodes, closed_nodes):
                new_tree.append([']', tree[node].label()])
                closed_nodes.append(node)
        for node in closed_nodes: 
            if node in open_nodes: open_nodes.remove(node)
    return new_tree

def close_check(node_list1, node_list2):
    for node in node_list1:
        if node not in node_list2: return False
    return True

def build_corpus(tree_bank, flat_structure = True):
    sentences = []
    for tree in tree_bank:
        #tree = subject_tag(tree)
        if flat_structure: sentence = [[x.leaves()[0], x.label()] for x in tree.subtrees() if len([y for y in x.subtrees()])==1]
        else: sentence = convert_tree(tree)
        sentence = filter_sentence(sentence)
        if sentence != 'bad-tag': sentences.append(sentence)
    kdata = []
    for sentence in sentences:
        for i in sentence.split():
            if i[len(i)-3:] in ['VBZ','VBP']: kdata.append(sentence); break
    return(kdata)


def filter_sentence(sentence):
    include = ['ROOT', 'FRAG', 'SBARQ', 'SBAR', 'SQ', 'S', 'SINV', 'WHNP', 'NP', 'VP', 'PRT', 'INTJ', 'WHPP', 'PP', 'WHADVP', 'ADVP', 'WHADJP', 'ADJP', 'NP-SUBJ', 
               'WP', 'NN', 'NNS', 'PRP', 'PRP$', 'CD', 'JJ', 'IN', 'VB', 'UH', 'TO', 'VBP', 'WRB', 'NOT', 'DT', 'RB', 'MD', 'RP', 'VBG', 'POS', 'VBZ', 
               'CC', 'VBD', 'COMP', 'EX', 'VBN', 'WDT', 'PDT', 'WP$', 'JJR', 'NN-SUBJ', 'NNS-SUBJ', 'PRP-SUBJ']
    exclude = [".", ",", ""]
    for i in sentence:
        # Regularize noun phrases (remove proper noun tags)
        if i[1] == 'NNP': i[1] = 'NN'
        if i[1] == 'NNPS': i[1] = 'NNS'
        # Label all non-fininte verbs VBP
        if i[1] == 'VB': i[1] = 'VBP'
        # Regularize copulas and BE axuiliaries
        if i[1] == 'COP' and i[0] == "'s": i[0] = 'BE'; i[1] = 'VBZ'
        if i[1] == 'COP' and i[0] == "'re": i[0] = 'BE'; i[1] = 'VBP'
        if i[0] in ['was', 'is']: i[0] = 'BE'; i[1] = 'VBZ'
        if i[0] in ['were', 'are']: i[0] = 'BE'; i[1] = 'VBP'
        # Regularize DO auxiliaries 
        if i[0] == 'does': i[0] = 'DO'; i[1] = 'VBZ'
        if i[0] == 'do': i[0] = 'DO'; i[1] = 'VBP'
        if i[0] == 'did': i[0] = 'DO'; i[1] = 'VBD'
        # Regularize HAVE auxiliaries
        if i[0] == 'has': i[0] = 'HAVE'; i[1] = 'VBZ'
        if i[0] == 'have': i[0] = 'HAVE'; i[1] = 'VBP'
        if i[0] == 'had': i[0] = 'HAVE'; i[1] = 'VBD'
        # If the sentence contains an uncrecognized tag, remove the sentence
        if i[1] not in include and i[1] not in exclude: return('bad-tag')
    # Return a string-version of the sentence
    return ' '.join(['/'.join(i) for i in sentence if i[1] not in exclude])

def get_pos_similarity(corpus):
    from math import log
    pos = ['ROOT', 'FRAG', 'SBARQ', 'SBAR', 'SQ', 'S', 'SINV', 'WHNP', 'NP', 'VP', 'PRT', 'INTJ', 'WHPP', 'PP', 'WHADVP', 'ADVP', 'WHADJP', 'ADJP', 'NP-SUBJ',
           'WP', 'NN', 'NNS', 'PRP', 'PRP$', 'CD', 'JJ', 'IN', 'VB', 'UH', 'TO', 'VBP', 'WRB', 'NOT', 'DT', 'RB', 'MD', 'RP', 'VBG', 'POS', 'VBZ',
           'CC', 'VBD', 'COMP', 'EX', 'VBN', 'WDT', 'PDT', 'WP$', 'JJR', 'NN-SUBJ', 'NNS-SUBJ', 'PRP-SUBJ']
    pos_frequency_dict, pos_similarity_dict = {}, {}
    # Fill in default values for frequency and similarity dictionaries
    for p in pos: 
        pos_frequency_dict[p], pos_similarity_dict[p] = {}, {}
        for p2 in pos:
            pos_frequency_dict[p][p2], pos_similarity_dict[p][p2] = 0.0000000001, 0
    # Loop over trees in corpus and, for each subtree, increment the value for the subtree's parent
    for tree in corpus:
        #tree = subject_tag(tree)
        tree = ParentedTree.convert(tree)
        for subtree in tree.subtrees():
            current_pos = subtree.label()
            parent_node = subtree.parent()
            if parent_node is not None and current_pos in pos and parent_node .label() in pos: pos_frequency_dict[current_pos][parent_node.label()] += 1
    # Loop over frequency dictionary, changing frequency counts to proportions
    for pos in pos_frequency_dict.keys():
        total = sum(pos_frequency_dict[pos].values())
        for pos2 in pos_frequency_dict[pos].keys(): 
            pos_frequency_dict[pos][pos2] = pos_frequency_dict[pos][pos2]/float(total)
    # Loop over entries in similiarity dictionary, calculating relative entropy for each category pair based on parent-node distributions
    for current_pos in pos_similarity_dict.keys():
        for compare_pos in pos_similarity_dict[current_pos].keys():
            #relative_entropy = []
            relative_entropy = 0
            for parent in pos_similarity_dict[current_pos].keys():
                p = pos_frequency_dict[current_pos][parent]
                q = pos_frequency_dict[compare_pos][parent]
                #relative_entropy.append(float(p)*log(float(p)/float(q), 2))
                relative_entropy += float(p)*log(float(p)/float(q), 2)
            #pos_similarity_dict[current_pos][compare_pos] = -sum(relative_entropy)
            pos_similarity_dict[current_pos][compare_pos] = -relative_entropy
        pos_similarity_dict[current_pos][current_pos] = 20
        pos_similarity_dict[current_pos]['VB____'] = -100
        pos_similarity_dict[current_pos]['*'] = 0
        pos_similarity_dict[current_pos]['XP'] = 0
    # Add in values for the gap position and the wild-card character
    pos_similarity_dict['VB____'] = {}
    pos_similarity_dict['*'] = {}
    pos_similarity_dict['XP'] = {}
    for pos in pos_similarity_dict.keys(): 
        pos_similarity_dict['VB____'][pos] = -100
        pos_similarity_dict['*'][pos] = 0
        pos_similarity_dict['XP'][pos] = 0
    pos_similarity_dict['VB____']['VB____'] = 100
    pos_similarity_dict['*']['*'] = 0
    pos_similarity_dict['XP']['XP'] = 0
    return pos_similarity_dict

def simulate(data, similarity_matrix, start = 50, max=50, by = 25, rep=5, conf=[0,.5], morphs = ['VBZ', 'VBP'], test=200, printR=False):
    import time
    from random import shuffle

    for x in range(0,rep):
        shuffle(data)
        n=start
        rules,contexts = {},{}
        while n <= max: 
            if n < max:
                grammar = generalize(data[0:n], similarity_matrix, rules, contexts, morphology = morphs, printRules = False)
                rules = grammar[0]
                contexts = grammar[1]
            else : rules = generalize(data[0:n], similarity_matrix, rules, contexts, morphology = morphs, printRules = printR, fileName = 'MGLgrammar.txt')[0]

            a = accuracy(data[n:n+test], rules, morphology = morphs, printAcc = printR, fileName = 'MGLresults.txt', similarity_matrix=similarity_matrix,trainSize=n,grammar=x)
            n+=by
