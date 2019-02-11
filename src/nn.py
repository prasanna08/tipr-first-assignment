import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
import utils

def classify(x, y, query, k):
	nearest_labels = y[np.argsort(np.linalg.norm(x - query, axis=1))[:k]]
	return Counter(nearest_labels).most_common()[0][0]

def test_accuracy(x, y, testx, testy, k):
	py = []
	for dx in testx:
		py.append(classify(x, y, dx, k))
	return utils.get_metrics(testy, np.array(py))

def sklearn_test_accuracy(x, y, testx, testy, k):
	clf = KNeighborsClassifier(k)
	clf.fit(x, y)
	return utils.get_metrics(testy, clf.predict(testx))
