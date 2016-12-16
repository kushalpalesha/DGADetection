#!env python

import pickle
import sys, re
from sklearn import tree
from feature_calculation_utils import get_features


# This script uses the combined decision tree model to classify an input domain

# Regular expression to verify domain name
domain = re.compile('^(?:[a-z0-9]+(-[a-z0-9]+)*\.)+[a-z]{2,}$')

if len(sys.argv) != 2:
    print "Invalid number of arguments."
    print "Usage: ./test_models_utils.py domain_name"

domain_name = sys.argv[1]

if domain.match(domain_name):
    model = pickle.load(open("../models/tree_models/combined.decision_tree","rb"))
    trigram_model = pickle.load(open("../models/ngram/trigram.model","rb"))
    quadgram_model = pickle.load(open("../models/ngram/quadgram.model","rb"))
    features = get_features(domain_name, [trigram_model, quadgram_model])
    result = model.predict(features)
    if result == 1:
        print domain_name + " was classified as an algorithimically generated domain"
    else:
        print domain_name + " was classified as a legitimiate domain"

else:
    print "Invalid domain name"
