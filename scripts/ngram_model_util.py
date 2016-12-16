import ngram
import pickle


def generate_ngram_model(listoffilenames, N):
    ngram_model = ngram.NGram(N=N)

    for filename in listoffilenames:
        with open(filename, "r") as text_file:
            for domain_name in text_file:
                second_level_domain = domain_name.split(".")[0]
                ngram_model.add(second_level_domain)

    return ngram_model

# trigram model
ngram_model = generate_ngram_model(["../domains/alexa_doNotUseForFeatureCalculation.txt"],N=3)
filename = "../models/ngram/trigram.model"
pickle.dump(ngram_model,open(filename,"wb"))

# quadgram model
ngram_model = generate_ngram_model(["../domains/alexa_doNotUseForFeatureCalculation.txt"],N=4)
filename = "../models/ngram/quadgram.model"
pickle.dump(ngram_model,open(filename,"wb"))
