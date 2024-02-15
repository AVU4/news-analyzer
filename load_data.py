import csv

import pandas as pd
from data_prep import preprocess


def save_data(filename, source, is_need_decode):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        for row in source:
            string = []
            for word in row:
                if is_need_decode:
                    decoded_word = word.decode('utf-8')
                    string.append(decoded_word)
                else:
                    string.append(word)
            writer.writerow(string)

if __name__ == '__main__' :

    df = pd.read_json('/home/avu/Загрузки/arxiv-metadata-oai-snapshot.json', lines=True, nrows=1000000)
    df = df.sample(frac=1).reset_index(drop=True)

    train_df = df[:int(0.8 * len(df))].reset_index(drop=True)
    test_df = df[int(0.8 * len(df)):].reset_index(drop=True)
    print(train_df.shape[0], 'training examples')
    print(test_df.shape[0], 'test examples')

    # Preprocess the data and labels for the train and test set.
    X_train = []
    y_train = []
    for c, (abstr, labs) in enumerate(zip(train_df['abstract'].tolist(), train_df['categories'].tolist())):
        X_train.append(preprocess(abstr))
        labs = labs.strip('[').strip(']').split(',')
        labs = [lab.strip() for lab in labs]
        y_train.append(labs)
        if c % 10000 == 0: print(c)
    X_test = []
    y_test = []
    for c, (abstr, labs) in enumerate(zip(test_df['abstract'].tolist(), test_df['categories'].tolist())):
        X_test.append(preprocess(abstr))
        labs = labs.strip('[').strip(']').split(',')
        labs = [lab.strip() for lab in labs]
        y_test.append(labs)
        if c % 10000 == 0: print(c)

    # Write the outputs to .csv
    print('Writting...')
    save_data('data/train_set.csv', X_train, True)
    save_data('data/test_set.csv', X_test, True)
    save_data('data/train_set_labels_small.csv', y_train, False)
    save_data('data/test_set_labels_small.csv', y_test, False)
