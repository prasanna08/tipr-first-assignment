from sklearn.metrics import f1_score, accuracy_score

def get_metrics(y_true, y_pred):
	return {
		'f1-micro': f1_score(y_true, y_pred, average='micro'),
		'f1-macro': f1_score(y_true, y_pred, average='macro'),
		'accuracy': accuracy_score(y_true, y_pred)
	}
