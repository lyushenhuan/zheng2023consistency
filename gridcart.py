from functools import reduce
from operator import mul
from itertools import product
import numpy as np

# GridCART for synthetic dataset
# The distribution is estimated by the product of estimation in each dimension.
class ProdGridCART:
	def __init__(self, N, max_depth=float('inf'), verbose=False):
		assert N is None or type(N) == int and N > 1, 'N should be an int > 1'

		self.N = N
		self.max_depth = max_depth
		self.verbose = verbose

	def normalize(self, X, Y):
		# Normalize Y to 0 and 1 binary value
		new_Y = ((Y - Y.min()) / (Y.max() - Y.min())).astype(int)

		# Normalize X to data in [0, 1] ^ d
		min_X = X.min(0)
		delta_X = X.max(0) - X.min(0)
		ratio = np.where(delta_X == 0, 0, 1 / np.where(delta_X == 0, 1, delta_X)) * (1 - 1e-5)
		new_X = (X - min_X) * ratio

		self.delta_X = min_X
		self.ratio = ratio

		return new_X, new_Y

	def histogram(self, X, Y):
		N = self.N
		n, d = X.shape

		marginal = np.zeros((d, N))
		grid_X = np.zeros((N,) * d, dtype=float)
		grid_Y = np.zeros((N,) * d, dtype=int)

		for (x, y) in zip(X, Y):
			for i in range(d):
				ind = min(int(x[i] * N), N - 1)
				marginal[i, ind] += 1
			grid_Y[tuple((x * N).astype(int).tolist())] += 2 * y - 1
		marginal /= n
		grid_X = reduce(mul, [marginal[i].reshape((1,) * i + (N,) + (1,) * (d - 1 - i)) for i in range(d)])
		grid_Y = (grid_Y >= 0).astype(int)
		return grid_X, grid_Y

	def generate_tree(self, X, Y, X_min, depth=0):
		assert(X.shape == Y.shape)
		# Stop splitting
		if Y.size == 0:
			return 0
		# The samme label
		if Y.max() - Y.min() == 0:
			return Y.min()
		# No cut point available
		if (np.array(X.shape) > 1).astype(int).sum() == 0:
			return 0
		if depth >= self.max_depth:
			return int(Y.sum() > 0.5)

		N = self.N
		i, s, _ = self.best_split(X, Y)
		# Update the hyperectanlge
		X_min_left = X_min[: ]
		X_min_right = X_min[: ]
		X_min_right[i] = s + X_min[i]

		left_ind = tuple([slice(0, _) for _ in X.shape[: i]]) + (slice(0, s),) + tuple([slice(0, _) for _ in X.shape[i + 1: ]])
		right_ind = tuple([slice(0, _) for _ in X.shape[: i]]) + (slice(s, X.shape[i]),) + tuple([slice(0, _) for _ in X.shape[i + 1: ]])

		if self.verbose:
			print('best split at {}, {}'.format(i, (s + X_min[i]) / N))
			print('left sample nums: {}'.format(X[left_ind].sum()))
			print('right sample nums: {}\n'.format(X[right_ind].sum()))
			
		tree = {
			'i': i,
			's': (s + X_min[i]) / N,
			'left': self.generate_tree(X[left_ind], Y[left_ind], X_min_left, depth + 1),
			'right': self.generate_tree(X[right_ind], Y[right_ind], X_min_right, depth + 1),
		}
		return tree

	def fit(self, X, Y, x_valid=None, y_valid=None):
		# Check data
		assert type(Y) == np.ndarray, 'Label "Y" must be a numpy array, but got a {}'.format(type(Y))
		assert len(Y.shape) == 1, 'Invalid label shape {}, it must be (n,)'.format(Y.shape)
		assert len(set(Y)) == 2, 'Too many the label values, got {} but should be binary value'.format(len(set(Y)))
		assert type(X) == np.ndarray, 'Input "X" must be a numpy array, but got a {}'.format(type(X))
		assert len(X.shape) == 2, 'Invalid label shape {}, it must be (n, d)'.format(X.shape)
		assert len(X) == len(Y), 'Length of X and Y mismatch'
		
		self.n, self.d = X.shape

		nor_X, nor_Y = self.normalize(X, Y)
		grid_X, grid_Y = self.histogram(nor_X, nor_Y)
		self.tree = self.generate_tree(grid_X, grid_Y, X_min=[0] * self.d)
	
	def best_split(self, X, Y):
		# return np.random.randint(0, 2), 1, None
		N = self.N
		d = len(X.shape)
		max_gain_dim, max_gain_k, max_gain = 0, 1, 0
		# Search all the dimensions
		for i in range(d):
			gain = 0
			for k in range(1, X.shape[i]):
				gain = 0
				lt = list(product(*[range(_) for _ in X.shape[: i]]))
				lt = lt if lt else [tuple()]
				gt = list(product(*[range(_) for _ in X.shape[i + 1: ]]))
				gt = gt if gt else [tuple()]

				for (x_lt_i, x_gt_i) in product(lt, gt):
					index = x_lt_i + (slice(0, X.shape[i]), ) + x_gt_i
					x = X[index]
					y = Y[index]

					if x.sum() == 0 or y.min() == y.max():
						continue

					left_ratio = x[: k].sum() / x.sum()
					right_ratio = 1 - left_ratio

					if left_ratio == 0 or right_ratio == 0:
						continue

					left_exp = (x[: k] * y[: k]).sum() / x[: k].sum()
					right_exp = (x[k: ] * y[k: ]).sum() / x[k: ].sum()
			
					gain += x.sum() * left_ratio * right_ratio * ((left_exp - right_exp) ** 2)

				if gain > max_gain:
					max_gain = gain
					max_gain_dim = i
					max_gain_k = k

		# No impurity gain at each cut point
		if max_gain == 0:
			for i in range(d):
				for k in range(1, X.shape[i]):
					left_index = tuple([range(_) for _ in X.shape[: i]]) + (slice(0, k), ) + tuple([range(_) for _ in X.shape[i + 1: ]])
					right_index = tuple([range(_) for _ in X.shape[: i]]) + (slice(k, X.shape[i]),) + tuple([range(_) for _ in X.shape[i + 1: ]])
					if X[left_index].sum() > 0 or X[right_index].sum() > 0:
						return i, k, 0
			raise ValueError('Some unknown error occurs')
		return max_gain_dim, max_gain_k, max_gain

	def _predict(self, x):
		assert type(x) == np.ndarray and len(x.shape) == 1, 'Input x should be a numpy array'
		def predict(tree, x):
			# Go through the whole tree
			if type(tree) == dict:
				if x[tree['i']] < tree['s']:
					return predict(tree['left'], x)
				else:
					return predict(tree['right'], x)
			else:
				return tree
		return predict(self.tree, x)

	def predict(self, X):
		# Predict and concat
		X = (X - self.delta_X) * self.ratio
		new_X = np.clip(X, 1e-5, 1 - 1e-5)
		return np.array([self._predict(x) for x in new_X])

	def get_depth(self):
		def depth(tree, n):
			if type(tree) != dict:
				return n
			l_depth = depth(tree['left'], n + 1)
			r_depth = depth(tree['right'], n + 1)
			return max(l_depth, r_depth)
		return depth(self.tree, 0)

	def get_n_leaves(self):
		def leaves(tree, s):
			if type(tree) != dict:
				return s + 1
			l_leaves = leaves(tree['left'], s)
			r_leaves = leaves(tree['right'], s)
			return l_leaves + r_leaves
		return leaves(self.tree, 0)

	def evaluate(self, X, Y):
		assert X.shape[1] == self.d and len(Y.shape) == 1 and len(X) == len(Y), 'Invalid input data'
		Y = ((Y - Y.min()) / (Y.max() - Y.min())).astype(int)
		output = self.predict(X)
		return (output == Y).mean()

	def show(self):
		from pprint import pformat
		return pformat(self.tree)

# GridCART for real-world datasets
# The distribution is estimated by kde directly on the joint distribution.
class GridCART:
	def __init__(self, N, max_depth=float('inf'), verbose=False):
		assert N is None or type(N) == int and N > 1, 'N should be an int > 1'

		self.N = N

		self.max_depth = max_depth

		self.verbose = verbose

	def normalize(self, X, Y):
		# Normalize Y to 0 and 1 binary value
		new_Y = ((Y - Y.min()) / (Y.max() - Y.min())).astype(int)

		# Normalize X to data in [0, 1] ^ d
		min_X = X.min(0)
		delta_X = X.max(0) - X.min(0)
		ratio = np.where(delta_X == 0, 0, 1 / np.where(delta_X == 0, 1, delta_X)) * (1 - 1e-5)
		new_X = (X - min_X) * ratio

		self.delta_X = min_X
		self.ratio = ratio

		return new_X, new_Y

	def histogram(self, X, Y):
		N = self.N

		grid = {}

		# Voting at each grid
				
		_X = (X * N).astype(int)
		for (x, y) in zip(_X, Y):
			p = str(x.tolist())
			grid[p] = grid.get(p, 0) + 2 * y - 1

		# Histogramize labels
		new_Y = np.copy(Y)
		for i in range(len(_X)):
			p = str(_X[i].tolist())
			new_Y[i] = 1 if grid[p] > 0 else 0
		return X, new_Y

	def generate_tree(self, X, Y, X_min, X_max, depth=0):
		# Stop splitting
		if len(Y) == 0:
			return 0
		# The samme label
		if Y.max() - Y.min() == 0:
			return Y[0]
		# No cut point available
		if sum(np.array(X_max) - np.array(X_min) <= 0) > 0:
			return 0
		if depth >= self.max_depth:
			return int(Y.sum() > 0.5)

		N = self.N
		i, s, _ = self.best_split(X, Y, X_min, X_max)
		# Update the hyperectanlge
		X_min_left = X_min
		X_max_left = X_max[: ]
		X_max_left[i] = s
		X_min_right = X_min[: ]
		X_min_right[i] = s
		X_max_right = X_max

		if self.verbose:
			print('best split at {}, {}'.format(i, s / N))
			print('left sample nums: {}'.format(len(Y[X[:, i] * N < s])))
			print('right sample nums: {}\n'.format(len(Y[X[:, i] * N >= s])))

		tree = {
			'i': i,
			's': s / N,
			'left': self.generate_tree(X[X[:, i] * N < s], Y[X[:, i] * N < s], X_min_left, X_max_left, depth + 1),
			'right': self.generate_tree(X[X[:, i] * N >= s], Y[X[:, i] * N >= s], X_min_right, X_max_right, depth + 1),
		}
		return tree

	def fit(self, X, Y, x_valid=None, y_valid=None):
		# Check data
		assert type(Y) == np.ndarray, 'Label "Y" must be a numpy array, but got a {}'.format(type(Y))
		assert len(Y.shape) == 1, 'Invalid label shape {}, it must be (n,)'.format(Y.shape)
		assert len(set(Y)) == 2, 'Too many the label values, got {} but should be binary value'.format(len(set(Y)))
		assert type(X) == np.ndarray, 'Input "X" must be a numpy array, but got a {}'.format(type(X))
		assert len(X.shape) == 2, 'Invalid label shape {}, it must be (n, d)'.format(X.shape)
		assert len(X) == len(Y), 'Length of X and Y mismatch'
		
		self.n, self.d = X.shape

		nor_X, nor_Y = self.normalize(X, Y)
		X, Y = self.histogram(nor_X, nor_Y)
		self.tree = self.generate_tree(X, Y, [0] * self.d, [self.N] * self.d)
	
	def best_split(self, X, Y, X_min, X_max):
		N = self.N
		n, d = X.shape
		gain = []

		# Search for all the dimensions
		for i in range(d):
			gain_i = []
			
			# (for each other dimensions, grid point, y and nums)
			_gain = np.zeros((n, self.N, 2))
			_X, _Y = (X * N).astype(int), Y
			sorted_X_Y = np.array(sorted(np.hstack((_X, np.expand_dims(_Y, 1))), key=lambda x: tuple((x[[_ for _ in range(d) if _ != i]]).astype(int).tolist())))
			_X = sorted_X_Y[:, : -1]
			_Y = sorted_X_Y[:, -1]
			last_x_i = _X[0][[x for x in range(d) if x != i]]
			index = 0
			for x, y in zip(_X, _Y):
				x_i = x[[x for x in range(d) if x != i]]
				xi = x[i]
				
				if (x_i != last_x_i).any():
					index += 1
				_gain[index][xi][0] += 2 * y - 1
				_gain[index][xi][1] += 1
				last_x_i = x_i
				
			_gain = _gain[_gain[:, :, 1].sum(1) > 0]
			# Possible cut point
			min_k = X_min[i]
			max_k = X_max[i]

			# Padding the gain list at first
			if min_k + 1 >= max_k:
				gain.append([0] * N)
				continue
			gain_i += [0] * min_k

			# Search for all possible cut points
			for k in range(min_k + 1, max_k):
				weight = _gain[:, :, 1].sum(1)

				left = _gain[:, :k, :]
				sum_left = left[:, :, 0].sum(1)
				num_left = left[:, :, 1].sum(1)
				ratio_left = num_left / weight
				exp_left = sum_left * np.where(num_left == 0, 0, 1 / np.where(num_left == 0, 1, num_left))

				right = _gain[:, k:, :]
				sum_right = right[:, :, 0].sum(1)
				num_right = right[:, :, 1].sum(1)
				ratio_right = num_right / weight
				exp_right = sum_right * np.where(num_right == 0, 0, 1 / np.where(num_right == 0, 1, num_right))

				gain_i.append((weight * ratio_left * ratio_right * (exp_left - exp_right) ** 2).sum())
			# Padding the gain list at last
			gain_i += [0] * int(N - max_k + 1)
			gain.append(gain_i)
		gain = np.array(gain)
		i = gain.max(1).argmax()
		s = gain[i].argmax() + 1

		# No impurity gain at each cut point
		if gain.max() == 0:
			for i in range(d):
				min_k = X_min[i]
				max_k = X_max[i]
				for k in range(min_k + 1, max_k):
					s = k
					left, right = X[X[:, i] * N < s], X[X[:, i] * N >= s]
					if len(left) < len(X) and len(right) < len(X):
						return i, k, gain
			raise ValueError('Some unknown error occurs')
		return i, s, gain

	def _predict(self, x):
		assert type(x) == np.ndarray and len(x.shape) == 1, 'Input x should be a numpy array'
		def predict(tree, x):
			# Go through the whole tree
			if type(tree) == dict:
				if x[tree['i']] < tree['s']:
					return predict(tree['left'], x)
				else:
					return predict(tree['right'], x)
			else:
				return tree
		return predict(self.tree, x)

	def predict(self, X):
		# Predict and concat
		X = (X - self.delta_X) * self.ratio
		new_X = np.clip(X, 1e-5, 1 - 1e-5)
		return np.array([self._predict(x) for x in new_X])

	def get_depth(self):
		def depth(tree, n):
			if type(tree) != dict:
				return n
			l_depth = depth(tree['left'], n + 1)
			r_depth = depth(tree['right'], n + 1)
			return max(l_depth, r_depth)
		return depth(self.tree, 0)

	def get_n_leaves(self):
		def leaves(tree, s):
			if type(tree) != dict:
				return s + 1
			l_leaves = leaves(tree['left'], s)
			r_leaves = leaves(tree['right'], s)
			return l_leaves + r_leaves
		return leaves(self.tree, 0)

	def evaluate(self, X, Y):
		assert X.shape[1] == self.d and len(Y.shape) == 1 and len(X) == len(Y), 'Invalid input data'
		Y = ((Y - Y.min()) / (Y.max() - Y.min())).astype(int)
		output = self.predict(X)
		return (output == Y).mean()

	def show(self):
		from pprint import pformat
		return pformat(self.tree)

if __name__ == '__main__':
	n, d = 100000, 6
	N = int(n ** (1 / (d + 2)))
	gc = GridCART(N=N)
	X = np.random.random((n, d))
	Y = ((X > 0.5).sum(1) % 2 == 0).astype(int)
	gc.fit(X, Y)
	print(gc.histogram(X, Y))