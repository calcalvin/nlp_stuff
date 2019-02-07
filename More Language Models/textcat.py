#!/usr/bin/env python

# Calvin Li
# cli78@jhu.edu
# 601.465 Natural Language Processing Assignment 3
# text-categorization
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

  if len(sys.argv) < 6 or (sys.argv[1] == 'TRAIN' and len(sys.argv) != 6):
    print("""
Prints the log-probability of each file under a smoothed n-gram model.

Usage:   {} TRAIN smoother lexicon trainpath1 trainpath2  
         {} TEST smoother lexicon trainpath1 trainpath2 prior files...
Example: {} TRAIN add1 words-10.txt gen spam
         {} TEST add1 words-10.txt gen spam 0.7 foo.txt bar.txt baz.txt

Possible values for smoother: uniform, add1, backoff_add1, backoff_wb, loglinear1
  (the \"1\" in add1/backoff_add1 can be replaced with any real lambda >= 0
   the \"1\" in loglinear1 can be replaced with any C >= 0 )
lexicon is the location of the word vector file, which is only used in the loglinear model
trainpath is the location of the training corpus
  (the search path for this includes "{}")
""".format(sys.argv[0], sys.argv[0], sys.argv[0], course_dir, sys.argv[0], course_dir, course_dir, Probs.DEFAULT_TRAINING_DIR))
    sys.exit(1)

  mode = sys.argv[1]
  argv = sys.argv[2:]

  smoother = argv.pop(0)
  lexicon = argv.pop(0)
  train_file1 = argv.pop(0)
  train_file2 = argv.pop(0)

  if mode == 'TEST':
    prior = float(argv.pop(0))
    if not argv:
        print("warning: no input files specified (missing prior)")

  # initialize language model
  def init_lm(corpus = None):
      lm = Probs.LanguageModel()
      lm.set_smoother(smoother)
      lm.read_vectors(lexicon)
      #   call lm.setVocabSize() on the pair of training corpora
      lm.set_vocab_size(train_file1, train_file2)

      return lm

  if mode == 'TRAIN':
    lm1 = init_lm()
    lm2 = init_lm()
    #  train model 1 from corpus 1
    lm1.train(train_file1)
    #  train model 2 from corpus 2
    lm2.train(train_file2)
    #  store the model parameters
    lm1.save(get_model_filename(smoother, lexicon, train_file1))
    lm2.save(get_model_filename(smoother, lexicon, train_file2))

  elif mode == 'TEST':
    if not argv:
      print("warning: no input files specified")
    #  restore model 1 and model 2 the previously saved parameters
    lm1 = Probs.LanguageModel.load(get_model_filename(smoother, lexicon, train_file1))
    lm2 = Probs.LanguageModel.load(get_model_filename(smoother, lexicon, train_file2))

    # We use natural log for our internal computations and that's
    # the kind of log-probability that fileLogProb returns.  
    # But we'd like to print a value in bits: so we convert
    # log base e to log base 2 at print time, by dividing by log(2).

    count1 = 0.
    count2 = 0.

    for testfile in argv:
      # log(p(train_file1 | w)) = log(p(w|train_file1)) + log(p(train_file1))
      logprob1 = lm1.filelogprob(testfile) + math.log(prior)
      # log(p(train_file2 | w)) = log(p(w|train_file2)) + log(p(train_file2))
      logprob2 = lm2.filelogprob(testfile) + math.log(1.0 - prior)

      if logprob1 > logprob2:
          count1 += 1
          print('%s   %s' % (train_file1, testfile))
      else:
          count2 += 1
          print('%s   %s' % (train_file2, testfile))

    percentage1 = count1 * 100.0 / (count1 + count2)
    percentage2 = 100.0 - percentage1
    print('%s files were more probably %s (%.2f%%)' % (count1, train_file1, percentage1))
    print('%s files were more probably %s (%.2f%%)' % (count2, train_file2, percentage2))
  else:
    sys.exit(-1)

if __name__ ==  "__main__":
  main()