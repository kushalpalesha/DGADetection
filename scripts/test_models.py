from test_models_utils import *

# Test the models on benign datasets
print "\n\nconficker scores --- "
print "opendns-random-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/conficker.decision_tree","../features/opendns-random-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/conficker.random_forest","../features/opendns-random-domains.npy"))
print "opendns-top-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/conficker.decision_tree","../features/opendns-top-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/conficker.random_forest","../features/opendns-top-domains.npy"))


print "\n\ncryptolocker scores: "
print "opendns-random-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/cryptolocker.decision_tree","../features/opendns-random-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/cryptolocker.random_forest","../features/opendns-random-domains.npy"))
print "opendns-top-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/cryptolocker.decision_tree","../features/opendns-top-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/cryptolocker.random_forest","../features/opendns-top-domains.npy"))


print "\n\nrovnix scores: "
print "opendns-random-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/rovnix.decision_tree","../features/opendns-random-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/rovnix.random_forest","../features/opendns-random-domains.npy"))
print "opendns-top-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/rovnix.decision_tree","../features/opendns-top-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/rovnix.random_forest","../features/opendns-top-domains.npy"))


print "\n\nmatsnu scores: "
print "opendns-random-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/matsnu.decision_tree","../features/opendns-random-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/matsnu.random_forest","../features/opendns-random-domains.npy"))
print "opendns-top-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/matsnu.decision_tree","../features/opendns-top-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/matsnu.random_forest","../features/opendns-top-domains.npy"))


print "\n\npushdo scores: "
print "opendns-random-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/pushdo.decision_tree","../features/opendns-random-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/pushdo.random_forest","../features/opendns-random-domains.npy"))
print "opendns-top-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/pushdo.decision_tree","../features/opendns-top-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/pushdo.random_forest","../features/opendns-top-domains.npy"))


print "\n\nramdo scores: "
print "opendns-random-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/ramdo.decision_tree","../features/opendns-random-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/ramdo.random_forest","../features/opendns-random-domains.npy"))
print "opendns-top-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/ramdo.decision_tree","../features/opendns-top-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/ramdo.random_forest","../features/opendns-top-domains.npy"))


print "\n\ntinba scores: "
print "opendns-random-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/tinba.decision_tree","../features/opendns-random-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/tinba.random_forest","../features/opendns-random-domains.npy"))
print "opendns-top-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/tinba.decision_tree","../features/opendns-top-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/tinba.random_forest","../features/opendns-top-domains.npy"))


print "\n\nzeus scores: "
print "opendns-random-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/zeus.decision_tree","../features/opendns-random-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/zeus.random_forest","../features/opendns-random-domains.npy"))
print "opendns-top-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/zeus.decision_tree","../features/opendns-top-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/zeus.random_forest","../features/opendns-top-domains.npy"))


# Test all the combined models on benign dataset
print "\n\ncombined scores: "
print "opendns-random-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/combined.decision_tree","../features/opendns-random-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/combined.random_forest","../features/opendns-random-domains.npy"))
print "SVM score: " + str(test_model("../models/tree_models/combined.svm","../features/opendns-random-domains.npy"))
print "K-means Prediction : " + str(kmeans_predict("../models/tree_models/combined.svm","../features/conficker.npy"))
print "opendns-top-domains scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/combined.decision_tree","../features/opendns-top-domains.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/combined.random_forest","../features/opendns-top-domains.npy"))
print "SVM score: " + str(test_model("../models/tree_models/combined.svm","../features/opendns-top-domains.npy"))
print "K-means Prediction : " + str(kmeans_predict("../models/tree_models/combined.svm","../features/conficker.npy"))



# Test the models on others_dga dataset
print "\n\nconficker scores --- "
print "others_dga scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/conficker.decision_tree","../features/others_dga.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/conficker.random_forest","../features/others_dga.npy"))


print "\n\ncryptolocker scores: "
print "others_dga scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/cryptolocker.decision_tree","../features/others_dga.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/cryptolocker.random_forest","../features/others_dga.npy"))


print "\n\nrovnix scores: "
print "others_dga scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/rovnix.decision_tree","../features/others_dga.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/rovnix.random_forest","../features/others_dga.npy"))


print "\n\nmatsnu scores: "
print "others_dga scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/matsnu.decision_tree","../features/others_dga.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/matsnu.random_forest","../features/others_dga.npy"))


print "\n\npushdo scores: "
print "others_dga scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/pushdo.decision_tree","../features/others_dga.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/pushdo.random_forest","../features/others_dga.npy"))


print "\n\nramdo scores: "
print "others_dga scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/ramdo.decision_tree","../features/others_dga.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/ramdo.random_forest","../features/others_dga.npy"))


print "\n\ntinba scores: "
print "others_dga scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/tinba.decision_tree","../features/others_dga.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/tinba.random_forest","../features/others_dga.npy"))


print "\n\nzeus scores: "
print "others_dga scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/zeus.decision_tree","../features/others_dga.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/zeus.random_forest","../features/others_dga.npy"))


print "others_dga scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/combined.decision_tree","../features/others_dga.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/combined.random_forest","../features/others_dga.npy"))




# Test combined models on all the agd datasets
print "\n\ncombined scores: "
print "conficker scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/combined.decision_tree","../features/conficker.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/combined.random_forest","../features/conficker.npy"))


print "\n\ncombined scores: "
print "cryptolocker scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/combined.decision_tree","../features/cryptolocker.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/combined.random_forest","../features/cryptolocker.npy"))


print "\n\ncombined scores: "
print "pushdo scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/combined.decision_tree","../features/pushdo.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/combined.random_forest","../features/pushdo.npy"))


print "\n\ncombined scores: "
print "ramdo scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/combined.decision_tree","../features/ramdo.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/combined.random_forest","../features/ramdo.npy"))


print "\n\ncombined scores: "
print "matsnu scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/combined.decision_tree","../features/matsnu.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/combined.random_forest","../features/matsnu.npy"))


print "\n\ncombined scores: "
print "rovnix scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/combined.decision_tree","../features/rovnix.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/combined.random_forest","../features/rovnix.npy"))


print "\n\ncombined scores: "
print "tinba scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/combined.decision_tree","../features/tinba.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/combined.random_forest","../features/tinba.npy"))


print "\n\ncombined scores: "
print "zeus scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/combined.decision_tree","../features/zeus.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/combined.random_forest","../features/zeus.npy"))


print "\n\ncombined scores: "
print "others_dga scores -- "
print "Decision Tree score: " + str(test_model("../models/tree_models/combined.decision_tree","../features/others_dga.npy"))
print "Random Forest score: " + str(test_model("../models/tree_models/combined.random_forest","../features/others_dga.npy"))



# Test combined models on all the agd datasets and the benign dataset together
print "\n\ncombined scores: "
print "conficker scores -- "
print "Decision Tree score: " + str(test("../models/tree_models/combined.decision_tree","../features/conficker.npy","../features/alexa.npy"))
print "Random Forest score: " + str(test("../models/tree_models/combined.random_forest","../features/conficker.npy","../features/alexa.npy"))


print "\n\ncombined scores: "
print "cryptolocker scores -- "
print "Decision Tree score: " + str(test("../models/tree_models/combined.decision_tree","../features/cryptolocker.npy","../features/alexa.npy"))
print "Random Forest score: " + str(test("../models/tree_models/combined.random_forest","../features/cryptolocker.npy","../features/alexa.npy"))


print "\n\ncombined scores: "
print "pushdo scores -- "
print "Decision Tree score: " + str(test("../models/tree_models/combined.decision_tree","../features/pushdo.npy","../features/alexa.npy"))
print "Random Forest score: " + str(test("../models/tree_models/combined.random_forest","../features/pushdo.npy","../features/alexa.npy"))


print "\n\ncombined scores: "
print "ramdo scores -- "
print "Decision Tree score: " + str(test("../models/tree_models/combined.decision_tree","../features/ramdo.npy","../features/alexa.npy"))
print "Random Forest score: " + str(test("../models/tree_models/combined.random_forest","../features/ramdo.npy","../features/alexa.npy"))


print "\n\ncombined scores: "
print "matsnu scores -- "
print "Decision Tree score: " + str(test("../models/tree_models/combined.decision_tree","../features/matsnu.npy","../features/alexa.npy"))
print "Random Forest score: " + str(test("../models/tree_models/combined.random_forest","../features/matsnu.npy","../features/alexa.npy"))


print "\n\ncombined scores: "
print "rovnix scores -- "
print "Decision Tree score: " + str(test("../models/tree_models/combined.decision_tree","../features/rovnix.npy","../features/alexa.npy"))
print "Random Forest score: " + str(test("../models/tree_models/combined.random_forest","../features/rovnix.npy","../features/alexa.npy"))


print "\n\ncombined scores: "
print "tinba scores -- "
print "Decision Tree score: " + str(test("../models/tree_models/combined.decision_tree","../features/tinba.npy","../features/alexa.npy"))
print "Random Forest score: " + str(test("../models/tree_models/combined.random_forest","../features/tinba.npy","../features/alexa.npy"))


print "\n\ncombined scores: "
print "zeus scores -- "
print "Decision Tree score: " + str(test("../models/tree_models/combined.decision_tree","../features/zeus.npy","../features/alexa.npy"))
print "Random Forest score: " + str(test("../models/tree_models/combined.random_forest","../features/zeus.npy","../features/alexa.npy"))


print "\n\ncombined scores: "
print "others_dga scores -- "
print "Decision Tree score: " + str(test("../models/tree_models/combined.decision_tree","../features/others_dga.npy","../features/alexa.npy"))
print "Random Forest score: " + str(test("../models/tree_models/combined.random_forest","../features/others_dga.npy","../features/alexa.npy"))



test_and_plot("../models/tree_models/combined.random_forest","../features/opendns-random-domains.npy","../features/opendns-top-domains.npy",
              "../features/others_dga.npy")
