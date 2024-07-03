import numpy as np
import random
from tqdm import tqdm
from data import *
from cart import MyCART
from gridcart import ProdGridCART, GridCART

def run(load_data, clf, times=20, seed=0):
	random.seed(seed)
	np.random.seed(seed)

	accs = []
	depths = []
	sizes = []
	for _ in tqdm(range(times)):
		(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data()
		clf.fit(x_train, y_train, x_valid, y_valid)
		accs.append(clf.evaluate(x_test, y_test) * 100)
		depths.append(clf.get_depth())
		sizes.append(clf.get_n_leaves())

	accs = np.array(accs)
	depths = np.array(depths)
	sizes = np.array(sizes)
	return {
		'acc': {'mean': accs.mean(), 'std': accs.std()},
		'depth': {'mean': depths.mean(), 'std': depths.std()},
		'size': {'mean': sizes.mean(), 'std': sizes.std()}
	}

if __name__ == '__main__':
	tasks = [
		{
			'name': 'Iris',
			'dataset': load_iris,
			'CART': MyCART(),
			'GridCART': GridCART(N=2),
		},
		{
			'name': 'Abalone',
			'dataset': load_abalone,
			'CART': MyCART(),
			'GridCART': GridCART(N=8),
		},
		{
			'name': 'Transfusion',
			'dataset': load_transfusion,
			'CART': MyCART(),
			'GridCART': GridCART(N=6),
		},
		{
			'name': 'Faults',
			'dataset': load_faults,
			'CART': MyCART(),
			'GridCART': GridCART(N=2),
		},
		{
			'name': 'Magic',
			'dataset': load_magic,
			'CART': MyCART(),
			'GridCART': GridCART(N=3),
		},
		{
			'name': 'Accelerometer',
			'dataset': load_accelerometer,
			'CART': MyCART(),
			'GridCART': GridCART(N=8),
		},
		{
			'name': 'Sin',
			'dataset': load_sin,
			'CART': MyCART(),
			'GridCART': ProdGridCART(N=6),
		},
		{
			'name': 'Ball',
			'dataset': load_ball,
			'CART': MyCART(),
			'GridCART': ProdGridCART(N=6),
		},
		{
			'name': 'Ring',
			'dataset': load_ring,
			'CART': MyCART(),
			'GridCART': ProdGridCART(N=6),
		},
		{
			'name': 'XOR',
			'dataset': load_xor,
			'CART': MyCART(),
			'GridCART': ProdGridCART(N=6),
		},
		{
			'name': 'Poly1',
			'dataset': load_poly1,
			'CART': MyCART(),
			'GridCART': ProdGridCART(N=6),
		},
		{
			'name': 'Poly2',
			'dataset': load_poly2,
			'CART': MyCART(),
			'GridCART': ProdGridCART(N=6),
		},
	]
	
	for task in tasks:
		print('#' * 80)
		print('task = {}'.format(task['name']))

		res_CART = run(task['dataset'], task['CART'])
		print('algo = CART')
		print('acc = {} +- {}'.format(res_CART['acc']['mean'], res_CART['acc']['std']))
		print('depth = {} +- {}'.format(res_CART['depth']['mean'], res_CART['depth']['std']))
		print('size = {} +- {}'.format(res_CART['size']['mean'], res_CART['size']['std']))

		res_GridCART = run(task['dataset'], task['GridCART'])
		print('algo = GridCART')
		print('acc = {} +- {}'.format(res_GridCART['acc']['mean'], res_GridCART['acc']['std']))
		print('depth = {} +- {}'.format(res_GridCART['depth']['mean'], res_GridCART['depth']['std']))
		print('size = {} +- {}'.format(res_GridCART['size']['mean'], res_GridCART['size']['std']))

		print('#' * 80 + '\n')