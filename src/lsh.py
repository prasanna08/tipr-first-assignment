# Implement code for Locality Sensitive Hashing here!
import numpy as np
from collections import defaultdict, Counter
import utils

def gen_hash(length, bucket_width=None, hashing_type='min'):
	if hashing_type == 'min':
		mapper = min_hash_mapper(length)
		return gen_min_hash(length, mapper)
	elif hashing_type == 'hamming':
		return gen_hamming_hash(length)
	elif hashing_type == 'e2lsh':
		assert bucket_width is not None, "E2LSH hash requires a bucket width"
		return gen_e2lsh_hash(length, bucket_width=bucket_width)

def gen_hamming_hash(length):
	c = np.random.choice(length, 1)[0]
	return lambda x: x[c]

def gen_hash_band(r, length, bucket_width=None, hashing_type='min'):
	b = [gen_hash(length, hashing_type=hashing_type, bucket_width=bucket_width) for _ in range(r)]
	return lambda x: [f(x) for f in b]

def min_hash_mapper(length):
	return np.random.choice(np.int(np.log2(length))**2, length)

def gen_min_hash(length, mapper):
	order = np.arange(length)
	np.random.shuffle(order)
	return lambda x: mapper[order[np.min(np.where(x[order] == 1))]]

def gen_e2lsh_hash(length, bucket_width):
	r = np.random.normal(size=(length,))
	b = np.random.uniform(bucket_width)
	return lambda x: np.round((np.dot(x, r) + b) / bucket_width)

def gen_bandfs(length, b, r, hashing_type='min', bucket_width=None):
	return [gen_hash_band(r, length, hashing_type=hashing_type, bucket_width=bucket_width) for _ in range(b)]

def init_bands(x, y, b, r, hashing_type='min', sep='|', bucket_width=None):
	bandfs = gen_bandfs(x.shape[1], b, r, hashing_type=hashing_type, bucket_width=bucket_width)
	bands = []
	for band in bandfs:
		item = {'bandfn': band, 'hashes': defaultdict(list)}
		for ind, i in enumerate(x):
			ihash = sep.join(str(j) for j in band(i))
			item['hashes'][ihash].append(ind)
		bands.append(item)
	return bands

def classify(q, y, bands, sep='|'):
		apx_neighbors = []
		missed_points = 0

		for band in bands:
			ihash = sep.join(str(j) for j in band['bandfn'](q))
			apx_neighbors.extend(band['hashes'][ihash])

		if not apx_neighbors:
			return
		res = Counter(y[np.unique(apx_neighbors)].tolist()).most_common()[0][0]
		return res

def test_accuracy(x, y, testx, testy, b, r, hashing_type='hamming', bucket_width=None):
	bands = init_bands(x, y, b, r, hashing_type=hashing_type, bucket_width=bucket_width)
	missed_points = 0
	py = []
	for q in testx:
		res = classify(q, y, bands)
		if not res:
			if missed_points == 0:
				print(
					'Warning: Some of the points might get missed because '
					'their hash doesn\'t match with hash of any other '
					'points in training data.')
			missed_points += 1
			py.append(-10)
			continue
		py.append(res)

	if missed_points > 0:
		print('Total %d points were missed during classification' % (missed_points))
		indices = np.where(np.array(py) != -10)
		return utils.get_metrics(np.array(testy)[indices], np.array(py)[indices])

	return utils.get_metrics(testy, py)

