import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:

	def __init__(self, size):
		self.size = size
		self.W, self.b = self.initialize_parameters()
		self.costs = []
		self.costs_dev = []

	def sigmoid(self, x):
		return (1 / (1 + np.exp(-x)))

	def relu(self, x):
		return (np.maximum(0, x))

	def sigmoid_back(self, x):
		return(x * (1 - x))

	def relu_back(self, x):
		x = x > 0
		return(x)

	def initialize_parameters(self):
		W = []
		b = []
		W.append(None)
		b.append(None)
		for l in range(1, len(self.size)):
			W.append(np.random.randn(self.size[l][0], self.size[l - 1][0]) * np.sqrt(2 / (self.size[l - 1][0])))
			# W.append(np.random.randn(self.size[l][0], self.size[l - 1][0]) / 100)
			b.append(np.zeros((self.size[l][0], 1)))
		return W, b

	def forward_propagate(self, X, W, b):
		Z = []
		A = []
		A.append(X)
		Z.append(None)
		for l in range(1, len(self.size)):
			Z.append(np.dot(W[l], A[l - 1]) + b[l])
			if (self.size[l][1] == "sigmoid"):
				A.append(self.sigmoid(Z[l]))
			elif(self.size[l][1] == "relu"):
				A.append(self.relu(Z[l]))
		return(Z, A)

	def compute_cost(self, X, Y, Y_hat, l2_reg):
		cost = (-1/X.shape[1]) * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
		add = 0
		for l in range (1, len(self.size)):
			add += np.sum(np.square(self.W[l]))
		cost += (l2_reg / (2 * X.shape[1])) * add
		return(cost)

	def back_propagate(self, X, Y, A, l2_reg):
		dZ = []
		dW = []
		db = []
		Y_hat = A[-1]
		dA = (Y_hat - Y) / (Y_hat * (1 - Y_hat))
		for l in reversed(range(1, len(self.size))):
			if (self.size[l][1] == "sigmoid"):
				dZ.insert(0 , dA * self.sigmoid_back(A[l]))
			elif (self.size[l][1] == "relu"):
				dZ.insert(0, dA * self.relu_back(A[l]))
			dW.insert(0, (np.dot(dZ[0], A[l-1].T) / X.shape[1]) + (l2_reg / X.shape[1]) * self.W[l])
			db.insert(0, np.sum(dZ[0], axis = 1, keepdims = True) / X.shape[1])
			dA = np.dot(self.W[l].T, dZ[0])
		dZ.insert(0, None)
		dW.insert(0, None)
		db.insert(0, None)
		return(dW, db)

	def update_parameters(self, learning_rate, dW, db):
		for l in range(1, len(self.size)):
			self.W[l] -= learning_rate * dW[l]
			self.b[l] -= learning_rate * db[l]

	def train(self, X, Y, X_test, Y_test, iterations, learning_rate, l2_reg):
		for i in range(0, iterations):
			Z, A = self.forward_propagate(X, self.W, self.b)
			cost = self.compute_cost(X, Y, A[-1], l2_reg)
			Z_test, A_test = self.forward_propagate(X_test, self.W, self.b)
			cost_dev = self.compute_cost(X_test, Y_test, A_test[-1], l2_reg)
			self.costs.append(cost.squeeze())
			self.costs_dev.append(cost_dev.squeeze())
			if (i == 0 or i % 50 == 49):
				print("Cost for iteration " + str(i + 1) + " = " + str(cost))
			dW, db = self.back_propagate(X, Y, A, l2_reg)
			self.update_parameters(learning_rate, dW, db)
			if (i % 25 == 0):
				plt.clf()
				plt.plot(self.costs)
				plt.plot(self.costs_dev)
				plt.ylabel("Cost")
				plt.xlabel("Iterations")
				plt.pause(0.000001)

	def predict(self, X, Y):
		Z, A = self.forward_propagate(X, self.W, self.b)
		Y_hat = A[-1]
		errors = np.sum(np.abs(Y - np.around(Y_hat)))
		success = X.shape[1] - errors
		print("Success: " + str(success) + " / " + str(X.shape[1]))
		print("Accuracy: " + str(success * 100 / X.shape[1]) + "%")

	def gch_params_to_vector(self, W, b):
		list = np.zeros((0,0))
		for l in range(1, len(self.size)):
			list = np.append(list, W[l].reshape(-1, 1))
		for l in range(1, len(self.size)):
			list = np.append(list, b[l].reshape(-1, 1))
		return(list.reshape(-1, 1))

	def gch_vector_to_params(self, list):
		W = []
		b = []
		W.append(None)
		b.append(None)
		start = 0
		for l in range(1, len(self.size)):
			end = start + self.W[l].shape[0] * self.W[l].shape[1]
			W.append(list[start:end].reshape(self.W[l].shape))
			start = end
		for l in range(1, len(self.size)):
			end = start + self.b[l].shape[0] * self.b[l].shape[1]
			b.append(list[start:end].reshape(self.b[l].shape))
			start = end
		return(W, b)

	def grd_check_n(self, X, Y, l2_reg = 0):
		Z, A = self.forward_propagate(X, self.W, self.b)
		Y_hat = A[-1]
		dW, db = self.back_propagate(X, Y, A, l2_reg)
		vector = self.gch_params_to_vector(self.W, self.b)
		vector2 = self.gch_params_to_vector(dW, db)
		num_parameters = vector.shape[0]
		print(vector.shape)
		gradapprox = np.zeros((num_parameters, 1))
		epsilon = 0.0000001
		for i in range(0, num_parameters):

			thetaplus = np.copy(vector)
			thetaplus[i][0] = thetaplus[i][0] + epsilon
			W_plus, b_plus = self.gch_vector_to_params(thetaplus)
			Z_plus, A_plus = self.forward_propagate(X, W_plus, b_plus)
			cost_plus = self.compute_cost(X, Y, A_plus[-1], l2_reg)

			thetaminus = np.copy(vector)
			thetaminus[i][0] = thetaminus[i][0] - epsilon
			W_minus, b_minus = self.gch_vector_to_params(thetaminus)
			Z_minus, A_minus = self.forward_propagate(X, W_minus, b_minus)
			cost_minus = self.compute_cost(X, Y, A_minus[-1], l2_reg)

			gradapprox[i] = (cost_plus - cost_minus) / (2 * epsilon)

		print(gradapprox)
		print(vector2)

		num = np.linalg.norm(vector2 - gradapprox)
		den = np.linalg.norm(vector2) + np.linalg.norm(gradapprox)
		diff = num / den

		print("Gradient check: " + str(diff))
