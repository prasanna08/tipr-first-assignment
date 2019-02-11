# Implement Bayes Classifier here!
import numpy as np
import utils

def get_class_conditional_probs(x, y, dist='Multinomial'):
	if dist in ('Multinomial', 'MultivariateBernoulli'):
		probs = np.ones(shape=(x.shape[1], np.unique(y).shape[0]))
	elif dist == 'Normal':
		mean = np.ones(shape=(x.shape[1], np.unique(y).shape[0]))
		var = np.ones(shape=(x.shape[1], np.unique(y).shape[0]))
	for i, c in enumerate(np.unique(y)):
		class_inputs = x[np.where(y == c)]
		if dist == 'Multinomial':
			probs[:, i] = (np.sum(class_inputs, axis=0) + 1) / float(np.sum(class_inputs) + class_inputs.shape[1])
		elif dist == 'MultivariateBernoulli':
			probs[:, i] = (np.sum(class_inputs > 0, axis=0) + 1) / float(class_inputs.shape[0] + class_inputs.shape[1])
		elif dist == 'Normal':
			mean[:, i] = np.mean(class_inputs, axis=0)
			var[:, i] = (np.var(class_inputs, axis=0) + 1e-10)
		else:
			raise Exception(
				'Unknown distribution. Please specify one of ' +
				'the Normal, Multinomial or MultivariateBernoulli.')

	return (mean, var) if dist == 'Normal' else probs

def classify(x, y, q, dist='Multinomial', probs=None, mean=None, var=None):
	if dist in ('Multinomial', 'MultivariateBernoulli') and (probs is None):
		probs = get_class_conditional_probs(x, y, dist=dist)
	elif dist == 'Normal' and (mean is None or var is None):
		mean, var = get_class_conditional_probs(x, y, dist=dist)

	p = {}
	for c, label in enumerate(np.unique(y)):
		p[label] = 1.0
		class_inputs = x[np.where(y == c)]
		if dist == 'Multinomial':
			locs = np.where(q > 0)
			p[label] = np.prod(probs[locs, c] ** q[locs])
		elif dist == 'MultivariateBernoulli':
			p[label] = np.prod(probs[np.where(q > 0), c]) * np.prod(1 - probs[np.where(q <= 0), c])
		elif dist == 'Normal':
			p[label] = np.prod(np.exp(-0.5 * ((q - mean[:, c]) ** 2) / var[:, c]) / np.sqrt(2 * np.pi * var[:, c]))
		else:
			raise Exception(
				'Unknown distribution. Please specify one of ' +
				'the Normal, Multinomial or MultivariateBernoulli.')

	return max(p.iteritems(), key=lambda x: x[1])[0]

def test_accuracy(x, y, testx, testy, dist='Multinomial'):
	py = []
	if dist in ('Multinomial', 'MultivariateBernoulli'):
		probs = get_class_conditional_probs(x, y, dist=dist)
		for q in testx:
			py.append(classify(x, y, q, dist=dist, probs=probs))
	elif dist == 'Normal':
		mean, var = get_class_conditional_probs(x, y, dist=dist)
		for q in testx:
			py.append(classify(x, y, q, dist=dist, mean=mean, var=var))
	return utils.get_metrics(testy, np.array(py))
