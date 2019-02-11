# Implement code for random projections here!
import numpy as np

def generate_proj_matrix(D, K):
	return np.random.uniform(size=(D, K))

def reduce_dim(data, K, proj=None):
	if proj is None:
		D = data.shape[1]
		proj = generate_proj_matrix(D, K)
	return np.dot(data, proj) / np.sqrt(K)
