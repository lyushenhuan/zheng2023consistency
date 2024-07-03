import pandas as pd
import numpy as np
import random
from copy import deepcopy

def split_train_test(data=None, X=None, Y=None, split_rate=(0.8, 0.1)):
	assert data is None or (X is None and Y is None), 'Please use \{data\} or both \{X\} and \{Y\}'
	assert type(X) == np.ndarray and type(Y) == np.ndarray, 'Please use numpy array for \{X\} and \{Y\}'

	# Normalize
	X = np.array(X).reshape(len(X), -1)
	Y = np.array(Y).reshape(-1)
	Y = ((Y - Y.min()) / (Y.max() - Y.min())).astype(int)

	# Gather data
	if data is None:
		data = list(zip(X, Y))
	_data = deepcopy(data)
	random.shuffle(_data)

	# Split train and test sets
	tr = int(len(_data) * split_rate[0])
	val = int(len(_data) * split_rate[1])
	train, valid, test = _data[: tr], _data[tr: tr + val], _data[tr + val: ]
	x_train, y_train = [p[0] for p in train], [p[1] for p in train]
	x_valid, y_valid = [p[0] for p in valid], [p[1] for p in valid]
	x_test, y_test = [p[0] for p in test], [p[1] for p in test]

	x_train, y_train = np.array(x_train), np.array(y_train)
	x_valid, y_valid = np.array(x_valid), np.array(y_valid)
	x_test, y_test = np.array(x_test), np.array(y_test)
	
	return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def load_abalone(split_rate=0.8):
	with open('datasets/abalone.data') as rf:
		raw = rf.read()
		raw = raw.replace('M', '0').replace('F', '1').replace('I', '2')
		data = raw.split('\n')[: -1]
		data = [[float(x) for x in row.split(',')] for row in data]
		X = np.array([row[: -1] for row in data])
		Y = np.array([row[-1] for row in data])
		Y = (Y < 10).astype(int)
		return split_train_test(X=X, Y=Y)

def load_iris(split_rate=0.8):
	with open('datasets/iris.data') as rf:
		raw = rf.read()
		raw = raw.replace('Iris-setosa', '0').replace('Iris-virginica', '1').replace('Iris-versicolor', '1')
		data = raw.split('\n')[: -1]
		data = [[float(x) for x in row.split(',')] for row in data]
		X = np.array([row[: -1] for row in data])
		Y = np.array([row[-1] for row in data])
		return split_train_test(X=X, Y=Y)

def load_magic(split_rate=0.8):
	with open('datasets/magic04.data') as rf:
		raw = rf.read()
		raw = raw.replace('g', '0').replace('h', '1')
		data = raw.split('\n')[: -1]
		data = [[float(x) for x in row.split(',')] for row in data]
		X = np.array([row[: -1] for row in data])
		Y = np.array([row[-1] for row in data])
		return split_train_test(X=X, Y=Y)

def load_transfusion(split_rate=0.8):
	with open('datasets/transfusion.data') as rf:
		raw = rf.read()
		data = raw.split('\n')[: -1]
		data = [[float(x) for x in row.split(',')] for row in data]
		X = np.array([row[: -1] for row in data])
		Y = np.array([row[-1] for row in data])
		return split_train_test(X=X, Y=Y)

def load_faults(split_rate=0.8):
	with open('datasets/Faults.NNA') as rf:
		raw = rf.read()
		data = raw.split('\n')[: -1]
		data = [[float(x) for x in row.split('\t')] for row in data]
		X = np.array([row[: -1] for row in data])
		Y = np.array([row[-1] for row in data])
		return split_train_test(X=X, Y=Y)

def load_accelerometer(split_rate=0.8):
	df = pd.read_csv('datasets/accelerometer.csv')
	X = df[['pctid', 'x', 'y', 'z']].to_numpy()
	Y = (df['wconfid'] <= 2).to_numpy().astype(int)
	return split_train_test(X=X, Y=Y)

def load_sin(split_rate=0.8, eps=0.05):
	X = np.random.random((10000, 4))
	Y = (X[:, 3] <= np.sin(X[:, :3].sum(1) * 5)).astype(int)
	Y = np.where(np.random.random(10000) < 0.05, 1 - Y, Y)
	return split_train_test(X=X, Y=Y)

def load_ball(split_rate=0.8, eps=0.05):
	X = np.random.random((10000, 4))
	Y = ((X ** 2).sum(1) < 1 / 2).astype(int)
	Y = np.where(np.random.random(10000) < 0.05, 1 - Y, Y)
	return split_train_test(X=X, Y=Y)

def load_ring(split_rate=0.8, eps=0.05):
	X = np.random.random((10000, 4))
	Y = ((X ** 2).sum(1) < 2 / 3).astype(int) * ((X ** 2).sum(1) > 1 / 3).astype(int)
	Y = np.where(np.random.random(10000) < 0.05, 1 - Y, Y)
	return split_train_test(X=X, Y=Y)	

def load_xor(split_rate=0.8, eps=0.05):
	X = np.random.random((10000, 4))
	Y = ((X > 0.5).sum(1) % 2 == 0).astype(int)
	Y = np.where(np.random.random(10000) < 0.05, 1 - Y, Y)
	return split_train_test(X=X, Y=Y)		

def load_poly1(split_rate=0.8, eps=0.05):
	X = np.random.random((10000, 4))
	Y = ((np.power(X, [1, 2, 3, 4]) * np.array([4, 3, 2, 1])).sum(1) < 4).astype(int)
	Y = np.where(np.random.random(10000) < 0.05, 1 - Y, Y)
	return split_train_test(X=X, Y=Y)

def load_poly2(split_rate=0.8, eps=0.05):
	X = np.random.random((10000, 4))
	Y = ((np.power(X, [4, 3, 2, 1]) * np.array([1, 2, 3, 4])).sum(1) < 4).astype(int)
	Y = np.where(np.random.random(10000) < 0.05, 1 - Y, Y)
	return split_train_test(X=X, Y=Y)	