import numpy as np
import argparse
from sklearn.feature_extraction.text import CountVectorizer
import nn as knn
import projections
import json
import bayes
import lsh

PARAMS = {
    'twitter': {
        'KNN': {'K': 55, 'D': 500},
        'LSH': {'B': 20, 'R': 5},
    },
    'pubmed': {
        'KNN': {'K': 50, 'D': 80},
        'LSH': {'B': 40, 'R': 5, 'bins': 6}
    },
    'dolphins': {
        'KNN': {'K': 5, 'D': 8},
        'LSH': {'B': 6, 'R': 6, 'bins': 2}
    }
}

def read_pubmed_dataset():
    x = np.genfromtxt('./data/pubmed/pubmed.csv', delimiter=' ')
    y = np.genfromtxt('./data/pubmed/pubmed_label.csv', delimiter='\n')
    return x[:5000], y[:5000]

def read_dolphins_dataset():
    x = np.genfromtxt('./data/dolphins/dolphins.csv', delimiter=' ')
    y = np.genfromtxt('./data/dolphins/dolphins_label.csv', delimiter='\n')
    return x, y

def read_twitter_dataset():
    with open('./data/twitter/twitter.txt', 'r') as f:
        corpus = list(f.readlines())
    labels = np.genfromtxt('./data/twitter/twitter_label.txt', delimiter='\n')
    return corpus, labels

def bag_of_words_transform(train, test):
    cv = CountVectorizer()
    train = cv.fit_transform(train).toarray()
    test = cv.transform(test).toarray()
    return train, test

def shuffle_data(x, y):
    shuffled_indices = np.arange(x.shape[0])
    np.random.shuffle(shuffled_indices)
    return x[shuffled_indices], y[shuffled_indices]

def data_split(x, y, cv=False):
    if cv:
        train = int(0.6 * x.shape[0])
        val = int(0.8 * x.shape[0])
        return [x[:train], y[:train]], [x[train: val], y[train: val]], [x[val:], y[val:]]
    else:
        train = int(0.8 * x.shape[0])
        return [x[:train], y[:train]], [x[train:], y[train:]]

def binarize_vectors(x):
    x[x >= 1] = 1
    return x

def find_best_k_and_d_for_knn(train, test):
    results = []
    for d in range(20, 100, 20):
        proj = projections.generate_proj_matrix(train[0].shape[1], d)
        x = projections.reduce_dim(train[0], d, proj)
        testx = projections.reduce_dim(test[0], d, proj)
        for k in [4, 10, 20, 35, 50, 65, 80]:
            print("Testing for K=%d, D=%d" % (k, d))
            scores = knn.test_accuracy(x, train[1], testx, test[1], k)
            print("Accuracy at K=%d, D=%d is %.4f" % (k, d, scores['accuracy']))
            results.append((k, d, scores))
    with open('pubmed-knn-output-result.log', 'w') as f:
        f.write(json.dumps(results))
    return results

def print_scores(score):
    print("Test accuracy :: %.3f" % score['accuracy'])
    print("Test F1-micro :: %.3f" % score['f1-micro'])
    print("Test F1-macro :: %.3f" % score['f1-macro'])

if __name__ == '__main__':
    print('Welcome to the world of high and low dimensions!')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data', type=str)
    parser.add_argument('--test-label', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    with open(args.test_data, 'r') as f:
        testx = np.genfromtxt(args.test_data, delimiter=' ')

    with open(args.test_label, 'r') as f:
        testy = np.genfromtxt(args.test_label, delimiter='\n')

    test = [testx, testy]

    if args.dataset.lower() == 'twitter':
        corpus, y = read_twitter_dataset()
        corpus, train[0] = bag_of_words_transform(corpus, test[0])
        train = [corpus, y]
        print("Running Bayes Classifier")
        print_scores(bayes.test_accuracy(train[0], train[1], test[0], test[1], dist='MultivariateBernoulli'))
        print("Running KNN with Random Projections")
        proj = projections.generate_proj_matrix(train[0].shape[1], PARAMS['twitter']['KNN']['D'])
        x = projections.reduce_dim(train[0], PARAMS['twitter']['KNN']['D'], proj)
        testx = projections.reduce_dim(test[0], PARAMS['twitter']['KNN']['D'], proj)
        print_scores(knn.test_accuracy(x, train[1], testx, test[1], PARAMS['twitter']['KNN']['K']))
        print("Running LSH with Hamming hash function family.")
        train[0] = binarize_vectors(train[0])
        test[0] = binarize_vectors(test[0])
        print_scores(lsh.test_accuracy(train[0], train[1], test[0], test[1], b=PARAMS['twitter']['LSH']['B'], r=PARAMS['twitter']['LSH']['R'], hashing_type='hamming'))

    elif args.dataset.lower() == 'pubmed':
        data, y = read_pubmed_dataset()
        train = [np.array(data), y]
        print("Running Bayes Classifier")
        print_scores(bayes.test_accuracy(train[0], train[1], test[0], test[1], dist='Normal'))
        print("Running KNN with Random Projections")
        proj = projections.generate_proj_matrix(train[0].shape[1], PARAMS['pubmed']['KNN']['D'])
        x = projections.reduce_dim(train[0], PARAMS['pubmed']['KNN']['D'], proj)
        testx = projections.reduce_dim(test[0], PARAMS['pubmed']['KNN']['D'], proj)
        print_scores(knn.test_accuracy(x, train[1], testx, test[1], PARAMS['pubmed']['KNN']['K']))
        print("Running LSH with E2LSH hash function family.")
        print_scores(lsh.test_accuracy(train[0], train[1], test[0], test[1], b=PARAMS['pubmed']['LSH']['B'], r=PARAMS['pubmed']['LSH']['R'], bucket_width=PARAMS['pubmed']['LSH']['bins'], hashing_type='e2lsh'))

    elif args.dataset.lower() == 'dolphins':
        data, y = read_dolphins_dataset()
        train = [np.array(data), y]
        print("Running Bayes Classifier")
        print_scores(bayes.test_accuracy(train[0], train[1], test[0], test[1], dist='Normal'))
        print("Running KNN with Random Projections")
        proj = projections.generate_proj_matrix(train[0].shape[1], PARAMS['pubmed']['KNN']['D'])
        x = projections.reduce_dim(train[0], PARAMS['pubmed']['KNN']['D'], proj)
        testx = projections.reduce_dim(test[0], PARAMS['pubmed']['KNN']['D'], proj)
        print_scores(knn.test_accuracy(x, train[1], testx, test[1], PARAMS['pubmed']['KNN']['K']))
        print("Running LSH with E2LSH hash function family.")
        print_scores(lsh.test_accuracy(train[0], train[1], test[0], test[1], b=PARAMS['pubmed']['LSH']['B'], r=PARAMS['pubmed']['LSH']['R'], bucket_width=PARAMS['pubmed']['LSH']['bins'], hashing_type='e2lsh'))