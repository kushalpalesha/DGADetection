import pickle
import ngram
from feature_calculation_utils import get_feature_array, combine_datasets
import numpy as np

trigram_model = pickle.load(open("../models/ngram/trigram.model","rb"))
quadgram_model = pickle.load(open("../models/ngram/quadgram.model","rb"))
feature_array = get_feature_array("../domains/alexa.txt", [trigram_model, quadgram_model])
np.save("../features/alexa", feature_array)

trigram_model = pickle.load(open("../models/ngram/trigram.model","rb"))
quadgram_model = pickle.load(open("../models/ngram/quadgram.model","rb"))
feature_array = get_feature_array("../domains/matsnu.txt", [trigram_model, quadgram_model])
np.save("../features/matsnu", feature_array)

trigram_model = pickle.load(open("../models/ngram/trigram.model","rb"))
quadgram_model = pickle.load(open("../models/ngram/quadgram.model","rb"))
feature_array = get_feature_array("../domains/zeus.txt", [trigram_model, quadgram_model])
np.save("../features/zeus", feature_array)

trigram_model = pickle.load(open("../models/ngram/trigram.model","rb"))
quadgram_model = pickle.load(open("../models/ngram/quadgram.model","rb"))
feature_array = get_feature_array("../domains/conficker.txt", [trigram_model, quadgram_model])
np.save("../features/conficker", feature_array)

trigram_model = pickle.load(open("../models/ngram/trigram.model","rb"))
quadgram_model = pickle.load(open("../models/ngram/quadgram.model","rb"))
feature_array = get_feature_array("../domains/cryptolocker.txt", [trigram_model, quadgram_model])
np.save("../features/cryptolocker", feature_array)

trigram_model = pickle.load(open("../models/ngram/trigram.model","rb"))
quadgram_model = pickle.load(open("../models/ngram/quadgram.model","rb"))
feature_array = get_feature_array("../domains/pushdo.txt", [trigram_model, quadgram_model])
np.save("../features/pushdo", feature_array)

trigram_model = pickle.load(open("../models/ngram/trigram.model","rb"))
quadgram_model = pickle.load(open("../models/ngram/quadgram.model","rb"))
feature_array = get_feature_array("../domains/ramdo.txt", [trigram_model, quadgram_model])
np.save("../features/ramdo", feature_array)

trigram_model = pickle.load(open("../models/trigram.model","rb"))
quadgram_model = pickle.load(open("../models/quadgram.model","rb"))
feature_array = get_feature_array("../domains/tinba.txt", [trigram_model, quadgram_model])
np.save("../features/tinba", feature_array)

# combine datasets
combine_datasets("combined",["conficker.npy","cryptolocker.npy","pushdo.npy","ramdo.npy",
                  "matsnu.npy","rovnix.npy","tinba.npy","zeus.npy"],100000)
