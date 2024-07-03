import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from data import *

# A CART that can automatically choose the best max_depth by cross-validation.
# The search will stop if the performance on the validation dataset has no improvement for 5 times.
class MyCART:
	def __init__(self):
		self.tree = None

	def fit(self, x_train, y_train, x_valid=None, y_valid=None, patience=5):
		tree = DecisionTreeClassifier()
		tree.fit(x_train, y_train)
		best_tree = tree

		# Search for the best max_depth by cross-validation
		if x_valid is not None and y_valid is not None:		
			max_acc = (tree.predict(x_valid) == y_valid).mean()
			wait = 0

			max_depth = tree.get_depth()
			while wait < patience and max_depth > 1:
				max_depth -= 1
				tree = DecisionTreeClassifier(max_depth=max_depth)
				tree.fit(x_train, y_train)
				acc = (tree.predict(x_valid) == y_valid).mean()
				if acc > max_acc:
					wait = 0
					max_acc = acc
					best_tree = tree
				else:
					wait += 1

		self.tree = best_tree

	def predict(self, x_test):
		assert self.tree is not None
		return self.tree.predict(x_test)

	def evaluate(self, x, y):
		assert self.tree is not None
		return (self.tree.predict(x) == y).mean()

	def get_depth(self):
		return self.tree.get_depth()

	def get_n_leaves(self):
		return self.tree.get_n_leaves()