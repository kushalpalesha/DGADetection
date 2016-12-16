import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.cluster import KMeans
import pickle

# Decision Tree
def decision_tree(agd_filename, legit_filename):
    agd_data = np.load(agd_filename)
    legit_data = np.load(legit_filename)

    # features L E G R
    agd_data = agd_data.T[:].T
    legit_data = legit_data.T[:].T

    agd_len = len(agd_data)
    agd_training_count = int(agd_len * 0.8)

    agd_training_set = agd_data[0:agd_training_count]
    agd_testing_set  = agd_data[agd_training_count:]

    legit_len = len(legit_data)
    legit_training_count = int(legit_len * 0.8)

    legit_training_set = legit_data[0:legit_training_count]
    legit_testing_set  = legit_data[legit_training_count:]

    labels = np.array(([1] * agd_training_count) + ([-1] * legit_training_count))

    training_set = np.append(agd_training_set, legit_training_set, axis=0)
    decistion_tree = DecisionTreeClassifier(criterion="entropy")
    model = decistion_tree.fit(training_set, labels)

    testing_set = np.append(agd_testing_set, legit_testing_set, axis=0)
    testing_labels = np.array(([1] * len(agd_testing_set)) + ([-1] * len(legit_testing_set)))

    print("Decision Tree mean accuracy:")
    print(model.score(testing_set, testing_labels))

    filename = "../models/tree_models/" + agd_filename.split("/")[-1][:-4] + ".decision_tree4"
    pickle.dump(model,open(filename,"wb"))



def randomforestclassifier(agd_filename, legit_filename):

    agd_data = np.load(agd_filename)
    legit_data = np.load(legit_filename)

    # features L E G R
    # agd_data = agd_data.T[2:].T
    # legit_data = legit_data.T[2:].T

    agd_len = len(agd_data)
    agd_training_count = int(agd_len * 0.8)

    agd_training_set = agd_data[0:agd_training_count]
    agd_testing_set = agd_data[agd_training_count:]

    legit_len = len(legit_data)
    legit_training_count = int(legit_len * 0.8)

    legit_training_set = legit_data[0:legit_training_count]
    legit_testing_set = legit_data[legit_training_count:]

    labels = np.array(([1] * agd_training_count) + ([-1] * legit_training_count))

    training_set = np.append(agd_training_set, legit_training_set, axis=0)
    decision_tree = RandomForestClassifier(criterion='entropy',bootstrap=True, n_jobs=-1,)
    model = decision_tree.fit(training_set, labels)


    testing_set = np.append(agd_testing_set, legit_testing_set, axis=0)
    testing_labels = np.array(([1] * len(agd_testing_set)) + ([-1] * len(legit_testing_set)))

    print("Random Forest mean accuracy:")
    print(model.score(testing_set, testing_labels))

    filename = "../models/tree_models/" + agd_filename.split("/")[-1][:-4] + ".random_forest4"
    pickle.dump(model, open(filename, "wb"))


def svm(agd_filename, legit_filename):

    legit_data = np.load(legit_filename)
    legit_data = legit_data[:1000]

    agd_data = np.load(agd_filename)

    agd_len = len(agd_data)
    agd_training_count = int(agd_len * 0.8)
    agd_training_set = agd_data[0:agd_training_count]
    agd_testing_set  = agd_data[agd_training_count:]

    legit_len = len(legit_data)
    legit_training_count = int(legit_len * 0.8)
    legit_training_set = legit_data[0:legit_training_count]
    legit_testing_set  = legit_data[legit_training_count:]


    (mean,sd) = calculate_mean_sd(agd_training_set,legit_training_set,agd_testing_set,legit_testing_set)


    training_set = np.append(agd_training_set, legit_training_set, axis=0)


    training_set = training_set.T
    training_array = []
    for i in range(0, len(training_set)):
        training_array.append(map(lambda x: (x - mean[i]) / sd[i], training_set[i]))

    train_set = np.array(training_array)
    training_set = train_set.T

    labels = np.array(([1] * agd_training_count) + ([-1] * legit_training_count))

    svm = SVC(C=1.5, kernel='rbf')
    model = svm.fit(training_set,labels)

    testing_set = np.append(agd_testing_set, legit_testing_set, axis=0)
    testing_set = testing_set.T
    testing_array = []
    for i in range(0, len(testing_set)):
        testing_array.append(map(lambda x: (x - mean[i]) / sd[i], testing_set[i]))

    test_set = np.array(testing_array)
    testing_set = test_set.T


    testing_labels = np.array(([1] * len(agd_testing_set)) + ([-1] * len(legit_testing_set)))

    print("SVM mean accuracy:")
    print(model.score(testing_set, testing_labels))

    filename = "../models/tree_models/" + agd_filename.split("/")[-1][:-4] + ".svm"
    pickle.dump(model, open(filename, "wb"))


# only training
def svm_combined(agd_filename, legit_filename):
    agd_data = np.load(agd_filename)
    legit_data = np.load(legit_filename)

    agd_len = len(agd_data)
    legit_len = len(legit_data)

    training_set = np.append(agd_data, legit_data, axis=0)
    trainingset_scaled = preprocessing.scale(training_set)

    labels = np.array(([1] * agd_len) + ([-1] * legit_len))

    svm = SVC(C=1.5, kernel='rbf')
    model = svm.fit(trainingset_scaled, labels)

    filename = "../models/tree_models/" + agd_filename.split("/")[-1][:-4] + ".svm"
    pickle.dump(model, open(filename, "wb"))


def calculate_mean_sd(agd_training_set,legit_training_set,agd_testing_set,legit_testing_set):
    mean = []
    sd = []

    training_set = np.append(agd_training_set, legit_training_set, axis=0)
    testing_set = np.append(agd_testing_set, legit_testing_set, axis=0)
    combined_set = np.append(training_set, testing_set, axis=0)

    length = len(combined_set)
    combined_set = combined_set.T

    for row in combined_set:
        mean.append(sum(row)/float(length))
        sd.append(np.std(row))

    return (mean,sd)



def combined_model(agd_filename, legit_filename):

    agd_data = np.load(agd_filename)
    legit_data = np.load(legit_filename)

    agd_len = len(agd_data)
    legit_len = len(legit_data)

    labels = np.array(([1] * agd_len) + ([-1] * legit_len))
    training_set = np.append(agd_data, legit_data, axis=0)

    random_forest = RandomForestClassifier(criterion='entropy',bootstrap=True, n_jobs=-1,)
    model = random_forest.fit(training_set, labels)
    filename = "../models/tree_models/" + agd_filename.split("/")[-1][:-4] + ".random_forest"
    pickle.dump(model, open(filename, "wb"))

    decision_tree = DecisionTreeClassifier(criterion="entropy")
    model = decision_tree.fit(training_set, labels)
    filename = "../models/tree_models/" + agd_filename.split("/")[-1][:-4] + ".decision_tree"
    pickle.dump(model, open(filename, "wb"))


def kmeans(agd_filename, legit_filename):
    agd_data = np.load(agd_filename)
    legit_data = np.load(legit_filename)

    agd_len = len(agd_data)
    print agd_len
    partition_size = int(agd_len/8)

    legit_len = len(legit_data)
    print legit_len

    labels = []
    for i in range(0,8):
        labels += ([i] * partition_size)

    labels = np.array(labels + ([8] * legit_len))
    print len(labels)
    print labels

    training_set = np.append(agd_data, legit_data, axis=0)


    kmeans =  KMeans(n_clusters=9)
    model = kmeans.fit(training_set, labels)
    filename = "../models/tree_models/" + agd_filename.split("/")[-1][:-4] + ".kmeans"
    pickle.dump(model, open(filename, "wb"))