import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import ndimage

db_size = 1000
db_size_test = 1000
img_size_x = 100
img_size_y = 100
Act = 0
iterations = 1000
learning_rate = 0.001
data_dims = img_size_x * img_size_y * 3
costs = []

def sigmoid(nbr):
	return(1 / (1 + np.exp(-nbr)))

def image_to_data(size, start):
	data = np.zeros((size, data_dims))
	print("Processing images...")
	for i in range(start, start + int(size / 2)):
		if (i % 50 == 0):
			print("Processed images: " + str(i * 2) + " / " + str(size))
		img = ndimage.imread("../train/cat." + str(i) + ".jpg")
		data[i - start] = sp.misc.imresize(img, (img_size_y, img_size_x)).reshape(data_dims)
		img = ndimage.imread("../train/dog." + str(i) + ".jpg")
		data[i + int(size / 2) - start] = sp.misc.imresize(img, (img_size_y, img_size_x)).reshape(data_dims)
	data = data / 255
	print(str(size) + " Images processed")
	return(data.T)

def set_labels(size):
	labels = np.zeros((1, size))
	for i in range(int(size / 2), size):
		labels[0][i] = 1
	return(labels)

def initialize_parameters():
	w = np.zeros((data_dims, 1))
	b = 0
	return(w, b)

def f_propagate(X_input, Y_lab, weights, bias, print_cost):
	Act = sigmoid(np.dot(weights.T, X_input) + bias)
	cost = np.sum(Y_lab * (np.log(Act)) + ((1 - Y_lab) *np.log(1 - Act))) / -db_size
	if (print_cost):
		print("Cost: " + str(cost))
	costs.append(cost)
	dw = np.dot(X_input ,(Act - Y_lab).T) / db_size
	db = np.sum(Act - Y_lab) / db_size
	return dw, db, Act

def optimize(weights, bias, dw, db, learning_rate):
	weights -= learning_rate * dw
	bias -= learning_rate * db
	return weights, bias

def training(X_input, Y_lab, weights, bias):
	print("Start training with " + str(iterations) + " iterations")
	for i in range (0, iterations):
		if (i % 100 == 0):
			dw, db, Act = f_propagate(X_input, Y_lab, weights, bias, 1)
		else:
			dw, db, Act = f_propagate(X_input, Y_lab, weights, bias, 0)
		weights, bias = optimize(weights, bias, dw, db, learning_rate)
	return weights, bias

def predict(X_input_test, Y_lab_test, weights, bias):
	dw, db, Act = f_propagate(X_input_test, Y_lab_test, weights, bias, 1)
	Act = np.around(Act)
	errors = np.sum(np.abs(Y_lab_test - Act))
	return(Y_lab_test.shape[1] - errors)

X_input = image_to_data(db_size, 0)
X_input_test = image_to_data(db_size_test, int(db_size / 2))
Y_lab = set_labels(db_size)
Y_lab_test = set_labels(db_size_test)
weights, bias = initialize_parameters()
weights, bias = training(X_input, Y_lab, weights, bias)
success = predict(X_input_test, Y_lab_test, weights, bias)
print("Success: " + str(success) + "/" + str(Y_lab_test.shape[1]))
costs.pop()
plt.plot(costs)
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.show()
