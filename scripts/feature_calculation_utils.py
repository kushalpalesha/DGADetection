import ngram
import string
import math
import numpy as np

def get_features(domain_name, ngram_model_list):
    second_level_domain = domain_name.split(".")[0]
    scores = []
    length = len(second_level_domain)
    scores.append(length)
    entropy = calculate_entropy(second_level_domain)
    scores.append(entropy)

    for ngram_model in ngram_model_list:
        ngram_score = sum([y for (_,y) in ngram_model.search(second_level_domain)])
        scores.append(ngram_score)

    return np.array(scores)

def calculate_ngram_score(ngram_model_filename, sample):
    ngram_model = pickle.load(open(ngram_model_filename,"rb"))
    result = ngram_model.search(sample)
    score = sum([y for (x, y) in result])
    return score

def calculate_entropy(sample):
    length = len(sample)
    characters = set(sample)
    entropy = 0
    for c in characters:
        p_x = float(sample.count(c))/length
        entropy += -p_x * math.log(p_x)
    return entropy

def calculate_alnum_continuity(sample):
    score = 0
    prev_char = ''
    alphabets = list(string.ascii_lowercase) + list('-')
    digits = list(string.digits)
    for x in sample:
        if prev_char == '':
            prev_char = x
        else:
            if (x in alphabets and prev_char in alphabets):
                continue
            elif (x in digits and prev_char in digits):
                continue
            else:
                score += 1
                prev_char = x
    score = float(score)/len(sample)
    return score

def get_feature_array(filename, ngram_model_list):
    temp_feature_array = []
    with open(filename, "r") as text_file:
        for domain_name in text_file:
            features = get_features(domain_name, ngram_model_list)
            print(".")
            temp_feature_array.append(features)
    return np.array(temp_feature_array)


def combine_datasets(fname, listoffilenames, datasetsize):
    partitionsize = int(datasetsize/len(listoffilenames))
    dataset = np.load("../features/"+listoffilenames[0])
    limited_dataset = dataset[0:partitionsize]
    combined_dataset = limited_dataset

    for filename in listoffilenames[1:]:
        dataset = np.load("../features/"+filename)
        limited_dataset = dataset[0:partitionsize]
        combined_dataset = np.append(combined_dataset,limited_dataset,axis=0)

    filepath = "../features/" + fname
    np.save(filepath, combined_dataset)
