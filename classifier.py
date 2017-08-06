import numpy as np
from math import sqrt, pi, exp


def guassian_prob(x, mu, sig):
	return exp(-(x - mu)**2/(2*sig**2))/sqrt(2*pi*sig**2)

class GNB(object):

	def __init__(self):
		self.possible_labels = ['left', 'keep', 'right']

	def train(self, data, labels):
		"""
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d, 
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
		"""
		dim = len(data[0])

		total_by_labels = {}
		for label in self.possible_labels:
			total_by_labels[label] = []
			# for i in range(dim):
			# 	total_by_labels[label].append([])

		print(total_by_labels)
		for X, label in zip(data, labels):
			# for i in range(dim):
			# 	total_by_labels[label][i].append(X[i])
			total_by_labels[label].append(X)

		means = []
		stds = []

		for label in self.possible_labels:
			# for i in range(dim):
			# 	mean = np.mean(total_by_labels[label][i])
			# 	print(mean)
			mean = np.mean(total_by_labels[label], axis=0)
			std = np.std(total_by_labels[label], axis=0)

			means.append(mean)
			stds.append(std)
		
		self._means = means
		self._stds = stds
		self.dim = dim


	def predict(self, observation):
		"""
		Once trained, this method is called and expected to return 
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
		probs = {}
		for (means, stds, label) in zip(self._means, self._stds, self.possible_labels):
			probs[label] = 1
			for (mu, sig, x) in zip(means, stds, observation):
				probs[label] *= guassian_prob(x, mu, sig)
			
        	return max(probs, key=probs.get)
	
	
