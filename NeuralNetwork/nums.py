import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from NeuralNetwork.SimpleNeuralNetwork import SimpleNeuralNetwork


db_size_train = 58000
db_size_test = 2000
img_x_size = 28
img_y_size = 28
img_data_size = img_x_size * img_y_size

img_fd = open('../nums/train-images.idx3-ubyte', 'rb')
img_fd.read(16)
lab_fd = open('../nums/train-labels.idx1-ubyte', 'rb')
lab_fd.read(8)

def img_to_data(size, start):
	X_local = np.zeros((size, img_data_size))
	for i in range(start, start + size):
		buff = img_fd.read(784)
		img = np.frombuffer(buff, dtype = np.uint8)
		X_local[i - start] = img
	return(X_local.T)

def define_labels(size, start):
	Y_local = np.zeros((1, size))
	for i in range(start, start + size):
		buff = lab_fd.read(1)
		Y_local[0][i - start] = ord(buff)
	return(Y_local)

def normalize(X, mean, standard_dev):
	X = (X - mean) / standard_dev
	return (X)

X = img_to_data(db_size_train, 0) / 255
# mean_x = np.sum(X, axis = 1, keepdims = True) / X.shape[1]
# std_dev = np.sqrt(np.sum(np.square(X - mean_x)) / X.shape[1]) + 1
# X= normalize(X, mean_x, std_dev)
Y = define_labels(db_size_train, 0) == 9

X_test = img_to_data(db_size_test, db_size_train) / 255
# X_test = normalize(X_test, mean_x, std_dev)
Y_test = define_labels(db_size_test, db_size_train) == 9

nn = SimpleNeuralNetwork([(img_data_size,), (12, "relu"), (12, "relu"), (10, "relu"), (4, "relu"), (1, "sigmoid")])
# nn.grd_check_n(Xx, Y)
nn.forward_propagate(X, nn.W, nn.b)
nn.train(X, Y, X_test, Y_test, iterations = 10000, learning_rate = 0.005, l2_reg = 0.1)
print("---------- Testing with training set --------------------")
nn.predict(X, Y)
print("---------- Testing with test set     --------------------")
nn.predict(X_test, Y_test)
plt.show()
