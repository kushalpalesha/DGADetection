import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import tree
import glob
import pydotplus

def test_model(modelfilename, test_features):
    model =  pickle.load(open(modelfilename,"rb"))
    test_set = np.load(test_features)
    test_set = test_set[:1000]

    # predictedList = model.predict(test_set)
    # falsepositives = np.ndarray.tolist(predictedList).count(1)
    # print "False Positives: " + str(falsepositives)

    return model.score(test_set, [1]*len(test_set))

def kmeans_predict(modelfilename, test_features):
    model =  pickle.load(open(modelfilename,"rb"))
    test_set = np.load(test_features)

    # print model.predict(test_set)
    return model.score(test_set, [0]*len(test_set))

def test(modelfilename,agd_filename,benign_filename):
    model = pickle.load(open(modelfilename, "rb"))
    agd_set = np.load(agd_filename)
    agd_set_len = len(agd_set)

    benign_set = np.load(benign_filename)[:agd_set_len]
    benign_set_len = len(benign_set)

    labels = [1] * agd_set_len + [-1] * benign_set_len

    test_set = np.append(agd_set,benign_set,axis=0)
    return model.score(test_set,labels)



def test_and_plot(modelfilename,opendns,topdns,others_dga):
    model = pickle.load(open(modelfilename, "rb"))

    opendns_test_set = np.load(opendns)[0:100]
    topdns_test_set = np.load(topdns)[0:100]
    others_dga_test_set = np.load(others_dga)[0:100]

    test_set = np.append(opendns_test_set,topdns_test_set,axis=0)
    test_set = np.append(test_set,others_dga_test_set,axis=0)

    opendns_len = len(opendns_test_set)
    topdns_len = len(topdns_test_set)
    others_dga_len = len(others_dga_test_set)

    labels = np.array([-1] * (opendns_len + topdns_len) + [1] * others_dga_len )
    color = np.array(['blue'] * opendns_len + ['yellow'] * topdns_len + ['red'] * others_dga_len)
    prediction = model.predict(test_set)

    print len(labels)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(range(0,len(color)), prediction, c=color, edgecolor = color)
    ax.set_xlabel('prediction')
    ax.set_ylabel('Datapoints')
    plt.xlim(-10, 320)
    plt.show()


def write_pdf(list_of_model_filenames):
    for modelfilename in list_of_model_filenames:
        print "."
        model = pickle.load(open(modelfilename,"rb"))
        trees = model.estimators_
        i = 0
        for treee in trees:
            model_tree = tree.export_graphviz(treee, feature_names=["length","entropy","tri-gram","quad-gram"] ,out_file=None)
            graph = pydotplus.graph_from_dot_data(model_tree)
            graph.write_pdf("/Users/tejassaoji/Desktop/Trees/RandomForest/combined/"+modelfilename.split("/")[-1]+"_"+str(i)+".pdf")
            i = i + 1


write_pdf(glob.glob("../models/tree_models/combined.random_forest"))