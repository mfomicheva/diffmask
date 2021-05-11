import pickle


def load_attributions(path):
    result = pickle.load(open(path, 'rb'))
    return result
