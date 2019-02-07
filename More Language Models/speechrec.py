#!/usr/bin/env python

# Sample program for hw-lm
# CS465 at Johns Hopkins University.

# Converted to python by Eric Perlman <eric@cs.jhu.edu>

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

from __future__ import print_function

import math
import sys

import Probs

# Computes the log probability of the sequence of tokens in file,
# according to a trigram model.  The training source is specified by
# the currently open corpus, and the smoothing method used by
# prob() is specified by the global variable "smoother". 

def get_model_filename(smoother, lexicon, train_file):
    import hashlib
    from os.path import basename
    train_hash = basename(train_file)
    lexicon_hash = basename(lexicon)
    filename = '{}_{}_{}.model'.format(smoother, lexicon_hash, train_hash)
    return filename

def main():
  course_dir = '/usr/local/data/cs465/'

  if len(sys.argv) < 5 or (sys.argv[1] == 'TRAIN' and len(sys.argv) != 5):
    print("""
Prints the log-probability of each file under a smoothed n-gram model.

Usage:   {} TRAIN smoother lexicon trainpath
         {} TEST smoother lexicon trainpath files...
Example: {} TRAIN add0.01 {}hw-lm/lexicons/words-10.txt switchboard-small
         {} TEST add0.01 {}hw-lm/lexicons/words-10.txt switchboard-small {}hw-lm/speech/sample*

Possible values for smoother: uniform, add1, backoff_add1, backoff_wb, loglinear1
  (the \"1\" in add1/backoff_add1 can be replaced with any real lambda >= 0
   the \"1\" in loglinear1 can be replaced with any C >= 0 )
lexicon is the location of the word vector file, which is only used in the loglinear model
trainpath is the location of the training corpus
  (the search path for this includes "{}")
""".format(sys.argv[0], sys.argv[0], sys.argv[0], course_dir, sys.argv[0], course_dir, course_dir, Probs.DEFAULT_TRAINING_DIR))
    sys.exit(1)

  argv = sys.argv[1:]

  smoother = argv.pop(0)
  lexicon = argv.pop(0)
  train_file = argv.pop(0)

  if not argv:
    print("warning: no input files specified")

  lm = Probs.LanguageModel()
  lm.set_smoother(smoother)
  lm.read_vectors(lexicon)
  lm.train(train_file)

  total_words = 0
  total_error = 0.0
  for testfile in argv:
    f = open(testfile)
    line = f.readline()
    sequences = []

    # Read data from file
    line = f.readline()
    while line:
      w_list = []
      items = line.split()
      line = f.readline()
      error_rate = float(items[0])
      log_p_uw = float(items[1])
      words = int(items[2])
      for i in range(3, words + 5):
        w_list.append(items[i])
      w_list = w_list[1:-1]

      # log probability computation
      # trigram model
      log_prob = 0.0
      x, y = Probs.BOS, Probs.BOS
      for z in w_list:
        log_prob += math.log(lm.prob(x, y, z))
        x = y
        y = z
      log_prob += math.log(lm.prob(x, y, Probs.EOS))

      # bigram model
      #y = Probs.BOS
      #for z in w:
      #  log_prob += math.log(lm.prob_bigram(y, z))
      #  y = z
      #log_prob += math.log(lm.prob_bigram(y, Probs.EOS))

      # unigram model
      #for z in w:
      #  log_prob += math.log(lm.prob_unigram(z))

      sequences.append((error_rate, words, log_p_uw + log_prob / math.log(2)))

    # Pick the best match, the one with highest probability
    best_match = max(sequences, key = lambda item : item[2])
    total_error += best_match[0] * best_match[1]
    total_words += best_match[1]
    print('{0}\t{1}'.format(best_match[0], testfile))

  print('{0:0.03f}\t{1}'.format(total_error / total_words, "OVERALL"))

if __name__ ==  "__main__":
  main()
