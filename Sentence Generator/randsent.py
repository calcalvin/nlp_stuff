#!/usr/bin/env python

"""
Random sentence generator that uses a CFG as an input file.
"""
import sys
import random
import argparse as ap

#Global Constants
NUM_SENTENCES = 1
START_SYMBOL = "ROOT"
M = 450
SENT_TREE = False
START_TREE = ""
TAB = "  "
SPACE = " "

def read_args():
    parser = ap.ArgumentParser(description='Command line options and arguments')
    parser.add_argument('-t', action='store_true', help="Print parse tree that represents underlying sentence structure")
    parser.add_argument('grmr', help="Grammar file path")
    parser.add_argument('num_sent', nargs='?',default=1, type=int, help="Number of sentences to be generated (Default: 1)")
    args = parser.parse_args()
    global NUM_SENTENCES, SENT_TREE
    if args.t:
        SENT_TREE = True
    if not args.num_sent:
        NUM_SENTENCES = 1
    else:
        NUM_SENTENCES = args.num_sent
    grammar_file = args.grmr
    return grammar_file

def read_grammar(grammar_file):
    fname = grammar_file
    with open(fname) as f:
        lines = f.readlines()
    #build dictionary of probabilities for rules
    probabilities = {}
    for line in lines:
        if (not line.startswith("#") and line !='\n'):
            tokens = line.split()
            if tokens[1] not in probabilities.keys():
                probabilities[tokens[1]] = float(tokens[0])
            else:
                probabilities[tokens[1]] += float(tokens[0])
    #build list for grammar
    grammar = [x.split() for x in lines if (not x.startswith("#") and x != "\n")]
    # skip in-line comments in grammar file
    for rule in grammar:
        if "#" in rule:
            rule_index = grammar.index(rule)
            comment_index = rule.index("#")
            grammar[rule_index] = rule[0:comment_index]
    return grammar, probabilities

def sentence_generator(symbol, grammar, sentence, prob_dict, sentence_tree, tree_format):
    #tree formatting
    if len(sentence_tree)>1:
        if sentence_tree[len(sentence_tree)-1]==")":
            sentence_tree.append("\n"+ tree_format)
        elif sentence_tree[len(sentence_tree)-1][3:] not in prob_dict.keys():
            sentence_tree.append("\n"+ tree_format)
    rule_prob = {}
    # we have reached the end of the recursion and can simply attach the last symbol
    if symbol not in prob_dict.keys():
        sentence.append(symbol)
        sentence_tree.append(TAB + symbol)
    elif "..." in symbol:
        sentence.append(symbol)
        sentence_tree.append(TAB + symbol + ")")
    else:
        sentence_tree.append(TAB + "(" + symbol)
        count = 0
        total = float(prob_dict[symbol])
        for rule in grammar:
            if rule[1] == symbol:
                count += float(rule[0])/total
                rule_prob[count] = rule
        #pick rule from probability and random number generator
        rand = random.random()
        next_rule = []
        s_rule_prob = sorted(rule_prob.keys())
        for p in s_rule_prob:
            if p >= rand:
                next_rule = rule_prob[p]
                #once rule is selected, we can begin recursive call
                break
        # terminates on any grammar
        branch_format = TAB + len(next_rule[1]) * SPACE
        if len(next_rule) > M:
            sentence_generator(symbol + "...", grammar, sentence, prob_dict, sentence_tree, tree_format + branch_format)
            return
        # Recurse to keep building the sentence
        # avoid RecursionError: maximum recursion depth exceeded in comparison
        for symbol in next_rule[2:len(next_rule)]:
            sentence_generator(symbol, grammar, sentence, prob_dict, sentence_tree, tree_format + branch_format)
        sentence_tree.append(")")

def generate(grammar, probabilities):
    sent_list = []
    sent_tree_list = []
    for i in range(int(NUM_SENTENCES)):
        sentence = []
        tree = []
        sentence_generator(START_SYMBOL, grammar, sentence, probabilities, tree, START_TREE)
        sent_list.append(' '.join(sentence))
        sent_tree_list.append(''.join(tree))
    for sent in sent_list:
        print(sent)
    if SENT_TREE:
        for tree in sent_tree_list:
            print(tree)

def main(arguments):
    g_file = read_args()
    grammar, prob = read_grammar(g_file)
    generate(grammar, prob)

if __name__ == "__main__":
    main(sys.argv[1:])
