import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from NeuralNetwork.SimpleNeuralNetwork import SimpleNeuralNetwork

np.random.seed(4000)

def img_to_data(size, start):
	X_local = np.zeros((size, img_data_size))
	for i in range(start, start + int(size / 2)):
		X_local[i - start] = sp.misc.imresize(ndi.imread("../train/cat." + str(i) + ".jpg"), (img_x_size, img_y_size)).reshape(1, img_data_size)
		X_local[i + int(size / 2) - start] = sp.misc.imresize(ndi.imread("../train/dog." + str(i) + ".jpg"), (img_x_size, img_y_size)).reshape(1, img_data_size)
		if (i == 0 or i % 50 == 0):
			print("Processing images: " + str((i - start) * 2) + " of " + str(size))
	print("------ 100% images processed ------")
	return(X_local.T)

def normalize(X, mean, standard_dev):
	X = (X - mean) / standard_dev
	return (X)

db_size_train = 1000
db_size_test = 100
img_x_size = 60
img_y_size = 60
img_data_size = img_x_size * img_y_size * 3



X = img_to_data(db_size_train, 0)
mean_x = np.sum(X, axis = 1, keepdims = True) / X.shape[1]
std_dev = np.sqrt(np.sum(np.square(X - mean_x)) / X.shape[1])
X= normalize(X, mean_x, std_dev)
Y = np.zeros((1, db_size_train))
for i in range(int(db_size_train / 2), db_size_train):
	Y[0][i] = 1

X_test = normalize(img_to_data(db_size_test, int(db_size_train / 2)), mean_x, std_dev)
Y_test = np.zeros((1, db_size_test))
for i in range(int(db_size_test / 2), db_size_test):
	Y_test[0][i] = 1

nn = SimpleNeuralNetwork([(img_data_size,), (7, "relu"), (3, "relu"), (3, "relu"), (1, "sigmoid")])
# nn.grd_check_n(Xx, Y)
nn.forward_propagate(X, nn.W, nn.b)
nn.train(X, Y, X_test, Y_test, iterations = 600, learning_rate = 0.1, l2_reg = 0.3)
print("---------- Testing with training set --------------------")
nn.predict(X, Y)
print("---------- Testing with test set     --------------------")
nn.predict(X_test, Y_test)
plt.show()
