#!/usr/bin/env python

"""
Calvin Li
cli78@jhu.edu
601.465 Natural Language Processing HW 2

Find similar words according to their word embeddings.
"""
import sys
import random
import math
import argparse as ap
import numpy as np

from numpy import dot
from numpy.linalg import norm

def read_args():
    if len(sys.argv[:]) == 3 or len(sys.argv[:]) == 5:
        pass
    else:
        print("Argument Error: usage: python findsim.py [in_file] [word] [word2_optional] [word3_optional]")
        return
    
    try:
        filename = sys.argv[1]
        target_word = sys.argv[2]
    except:
        print("Invalid file: usage: python findsim.py [in_file] [word] [word2_optional] [word3_optional]")

    try:
        word_2 = sys.argv[3]
        word_3 = sys.argv[4]
    except:
        word_2 = None
        word_3 = None
    
    return filename, target_word, word_2, word_3

def read_w2v(w2v_file):
    fname = w2v_file
    with open(fname) as f:
        lines = f.readlines()[1:]
    f.close()
    #build dictionary of probabilities for rules
    word_vectors = {}
    for line in lines:
        tokens = line.split()
        if tokens[0] not in word_vectors:
            int_vec = list(map(float, tokens[1:]))
            word_vectors[tokens[0]] = np.asarray(int_vec)
        
    return word_vectors

def cosine_similarity(v,w):
    """compute cosine similarity of v to w: (v dot w)/{||v||*||w||)"""
    cos_sim = dot(v, w)/(norm(v)*norm(w))
    return cos_sim

def find_top_10(target_vec, word_vectors, target_words):
    most_similar = {}
    for word in word_vectors:
        if word in target_words:
            continue
        word_vec = word_vectors[word]

        cos_sim = cosine_similarity(target_vec, word_vec)
        if len(most_similar) < 10:
            most_similar[word] = cos_sim
        else:
            min_key = min(most_similar, key = most_similar.get)
            if cos_sim > most_similar[min_key]:
                del most_similar[min_key]
                most_similar[word] = cos_sim
    return most_similar

def main(arguments):
    w2v_file, target_word, word2, word3 = read_args()
    word_vectors = read_w2v(w2v_file)
    target_vec = word_vectors[target_word]
    target_words = {target_word}

    if word2 is None:
        top_10 = find_top_10(target_vec, word_vectors, target_words)
    else:
        target_words = {target_word, word2, word3}
        target_vec = word_vectors[target_word] - word_vectors[word2] + word_vectors[word3]
        top_10 = find_top_10(target_vec, word_vectors, target_words)
    
    sorted_top_10 = sorted(top_10.items(), key=lambda kv: kv[1], reverse=True)
    sorted_top_10.append(("\n", 0))
    for i in sorted_top_10:
        print(i[0], end=" ")

if __name__ == "__main__":
    main(sys.argv[1:])
