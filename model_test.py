import sklearn.metrics as metrics

from classifier import load_data_test
from classifier import load_model as lm
from data_gen import Corpus
import numpy as np

def get_y_values(X_test):
    predictions = model.predict(X_test)

    y_test_prediction = []
    for prediction in predictions:
        max_class = prediction.argmax()
        y_test_prediction.append(max_class)
    return y_test_prediction

if __name__ == "__main__":

    model = lm('versions/v1/doc_blstm')

    DATA_DIR = '/home/avu/Pycharm/news-analyzer/models/blstm/data/'
    corpus = Corpus(DATA_DIR + 'test_set.csv', DATA_DIR + 'test_set_labels_small.csv')

    X_test, y_test = load_data_test(corpus)
    X_test[np.isnan(X_test)] = 0

    array_split = np.array_split(X_test, 2)
    first_part = array_split[0]
    second_part = array_split[1]
    first_part = get_y_values(first_part)
    second_part = get_y_values(second_part)
    y_test_prediction = np.concatenate((first_part, second_part))

    y_test_original = []
    for y in y_test:
        y_test_original.append(y.argmax())

    accuracy_score = metrics.accuracy_score(y_test_original, y_test_prediction)

    print('Точность = ' + str(accuracy_score))



