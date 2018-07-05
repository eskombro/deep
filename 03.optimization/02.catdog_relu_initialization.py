import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import ndimage as ndi

np.random.seed(42)

db_size = 1000
iterations =  5000
db_size_test = 1000
img_x_size = 60
img_y_size = 60
img_data_size = img_x_size * img_y_size * 3
learning_rate = 0.01

n_x = 0
n_h = 6
n_y = 0
W = 0
b = 0

def sigmoid(x):
	return(1 / (1 + np.exp(-x)))

def relu(x):
	return(np.maximum(0, x))

def img_to_data(size, start):
	X_local = np.zeros((size, img_data_size))
	for i in range(start, start + int(size / 2)):
		X_local[i - start] = sp.misc.imresize(ndi.imread("../train/cat." + str(i) + ".jpg"), (img_x_size, img_y_size)).reshape(1, img_data_size)
		X_local[i + int(size / 2) - start] = sp.misc.imresize(ndi.imread("../train/dog." + str(i) + ".jpg"), (img_x_size, img_y_size)).reshape(1, img_data_size)
		if (i == 0 or i % 50 == 0):
			print("Processing images: " + str((i - start) * 2) + " of " + str(size))
	print("------ 100% images processed ------")
	return(X_local.T / 255)

def set_labels(size):
	Y = np.zeros((1, size))
	for i in range(0, int(size / 2)):
		Y[0][i] = 1;
	return (Y)

def initialize_parameters(W, b):
	W1 = np.random.randn(n_h, n_x) * np.sqrt(2 / n_x)
	W2 = np.random.randn(n_y, n_h) * np.sqrt(2 / n_h)
	b1 = np.zeros((n_h, 1))
	b2 = np.zeros((n_y, 1))
	return {"W1": W1, "W2": W2, "b1": b1, "b2": b2}

def forward_propagation(X, params):
	Z1 = np.dot(params["W1"], X) + params["b1"]
	A1 = relu(Z1)
	Z2 = np.dot(params["W2"], A1) + params["b2"]
	A2 = sigmoid(Z2)
	return {"Z1": Z1, "Z2": Z2, "A1": A1, "A2": A2}

def calculate_cost(cache):
	cost = (-1/db_size) * np.sum(np.dot(Y, np.log(cache["A2"]).T) + np.dot((1 - Y), np.log(1 - cache["A2"]).T))
	return(cost)

def back_propagation(X, cache, params):
	dZ2 = cache["A2"] - Y
	dW2 = np.dot(dZ2, cache["A1"].T) / db_size
	db2 = np.sum(dZ2, axis = 1, keepdims = True) / db_size
	dZ1 = np.dot(params["W2"].T, dZ2) * (np.ceil((-1 / (1 + cache["A1"])) + 1))
	dW1 = np.dot(dZ1, X.T) / db_size
	db1 = np.sum(dZ1, axis = 1, keepdims = True) / db_size
	return {"dW1": dW1, "dW2": dW2, "db1": db1, "db2": db2}

def update_parameters(params, grads, learning_rate):
	params["b1"] = params["b1"] - learning_rate * grads["db1"]
	params["b2"] = params["b2"] - learning_rate * grads["db2"]
	params["W1"] = params["W1"] - learning_rate * grads["dW1"]
	params["W2"] = params["W2"] - learning_rate * grads["dW2"]
	return(params)

def training(X, iterations, learning_rate):
	params = initialize_parameters(W, b)
	last_cost = 10.0
	for i in range(0, iterations):
		cache = forward_propagation(X, params)
		cost = calculate_cost(cache)
		# if (cost >= last_cost):
		# 	learning_rate *= 0.9
		# 	print("Learning rate modified at " + str(i) + ": " + str(learning_rate))
		last_cost = cost
		if (i == 0 or i % 100 == 99):
			print("Cost at iteration " + str(i) + ": " + str(cost))
		grads = back_propagation(X, cache, params)
		params = update_parameters(params, grads, learning_rate)
	return(params)

def predict(X_test, Y_test, params):
	result = forward_propagation(X_test, params)
	errors = np.sum(np.abs(Y_test - np.around(result["A2"])))
	success = int(X_test.shape[1] - errors)
	print("Success: " + str(success) + "/" + str(int(X_test.shape[1])))
	print("Succes rate: " + str((success * 100) / X_test.shape[1]))

X = img_to_data(db_size, 0)
Y = set_labels(db_size)
X_test = img_to_data(db_size_test, int(db_size / 2))
Y_test = set_labels(db_size_test)
n_x = X.shape[0]
n_y = Y.shape[0]
params = training(X, iterations, learning_rate)
print("--- Predict sur training examples in db ---")
predict(X, Y, params)
print("--- Predict sur test ----------------------")
predict(X_test, Y_test, params)
