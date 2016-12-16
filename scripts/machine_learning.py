from machine_learning_utils import *

decision_tree("../features/rovnix.npy","../features/alexa.npy")
randomforestclassifier("../features/rovnix.npy","../features/alexa.npy")

decision_tree("../features/matsnu.npy","../features/alexa.npy")
randomforestclassifier("../features/matsnu.npy","../features/alexa.npy")

decision_tree("../features/pushdo.npy","../features/alexa.npy")
randomforestclassifier("../features/pushdo.npy","../features/alexa.npy")

decision_tree("../features/ramdo.npy","../features/alexa.npy")
randomforestclassifier("../features/ramdo.npy","../features/alexa.npy")

decision_tree("../features/conficker.npy","../features/alexa.npy")
randomforestclassifier("../features/conficker.npy","../features/alexa.npy")

decision_tree("../features/cryptolocker.npy","../features/alexa.npy")
randomforestclassifier("../features/cryptolocker.npy","../features/alexa.npy")

decision_tree("../features/tinba.npy","../features/alexa.npy")
randomforestclassifier("../features/tinba.npy","../features/alexa.npy")

decision_tree("../features/zeus.npy","../features/alexa.npy")
randomforestclassifier("../features/zeus.npy","../features/alexa.npy")

combined_model("../features/combined.npy","../features/alexa.npy")
svm_combined("../features/combined.npy","../features/alexa.npy")
kmeans("../features/combined.npy","../features/alexa.npy")