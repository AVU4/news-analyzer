import sklearn.metrics as metrics

from classifier import load_data_test
from classifier import load_model as lm
from data_gen import Corpus

if __name__ == "__main__":

    model = lm('doc_blstm')
    model.load_weights('/home/avu/Pycharm/news-analyzer/models/blstm/doc_blstm.h5')

    DATA_DIR = '/home/avu/Pycharm/news-analyzer/models/blstm/data/'
    corpus = Corpus(DATA_DIR + 'test_set.csv', DATA_DIR + 'test_set_labels_small.csv')

    X_test, y_test = load_data_test(corpus)

    predictions = model.predict(X_test)

    y_test_prediction = []
    for prediction in predictions:
        max_class = prediction.argmax()
        y_test_prediction.append(max_class)

    y_test_original = []
    for y in y_test:
        y_test_original.append(y.argmax())

    print(y_test_prediction)
    print(y_test_original)

    accuracy_score = metrics.accuracy_score(y_test_original, y_test_prediction)

    print('Точность = ' + str(accuracy_score))


