import numpy as np


class Utils:
	@staticmethod
	def relu(x):
		return np.maximum(0, x)

	@staticmethod
	def relu_derivative(x):
		return (x > 0).astype(np.float32)

	@staticmethod
	def softmax(x):
		x = x - np.max(x)
		exp = np.exp(x)
		return exp / np.sum(exp)


class NeuralNetwork:
	def __init__(self, layer_info):
		# layer_info : [inputs, hidden1, hidden2 , output]
		self.layer_info = layer_info
		self.L = len(layer_info) - 1

		self.W = [np.random.randn(layer_info[l], layer_info[l-1]) * np.sqrt(2/layer_info[l-1]) for l in range(1, len(layer_info))]
		self.B = [np.ones((layer_info[l], )) for l in range(1, len(layer_info))]
	
	def forward(self, inputs):
		A = [inputs]
		Z = []
		for l in range(self.L - 1):
			z = self.W[l] @ A[-1] + self.B[l]
			Z.append(z)

			A.append(Utils.relu(z))
		# for output layer softmax
		Z.append(self.W[-1] @ A[-1] + self.B[-1])
		A.append(Utils.softmax(Z[-1]))


		return Z, A
	
	def train_sgd(self, X, Y, X_test, Y_test, epochs = 100, lr = 0.001):
		n = len(X)

		for epoch in range(epochs):

			for i in range(n):
				Z, A = self.forward(X[i])
				
				dz = A[-1] - Y[i]

				for l in reversed(range(self.L)):
					self.B[l] -= lr * dz
					dw = self.W[l]
					self.W[l] -= lr * np.outer(dz, A[l])

					if l > 0:
						dz = (dw.T @ dz) * Utils.relu_derivative(Z[l-1])
			correct = 0

			n = len(X_test)

			for i in range(n):
				Z,A = self.forward(X_test[i])
				pred = np.argmax(A[-1])
				real = np.argmax(Y_test[i])
				if pred == real:
					correct +=1
			
			print("Epoch: ", epoch, ", Accuracy: ", (correct/n) * 100)

	def train_batch(self, X, Y, X_test, Y_test, epochs = 100, lr = 0.001):
		n = len(X)

		for epoch in range(epochs):
			dW = [np.zeros((self.layer_info[l], self.layer_info[l-1])) for l in range(1, len(self.layer_info))]
			dB = [np.zeros((self.layer_info[l], )) for l in range(1, len(self.layer_info))]

			for i in range(n):
				Z, A = self.forward(X[i])
				
				dz = A[-1] - Y[i]

				for l in reversed(range(self.L)):
					dW[l] += np.outer(dz, A[l])
					dB[l] += dz
					

					if l > 0:
						dz = (self.W[l].T @ dz) * Utils.relu_derivative(Z[l-1])
			for l in range(self.L):
				self.W[l] -= lr * dW[l] / n
				self.B[l] -= lr * dB[l]	/ n
			correct = 0

			n = len(X_test)

			for i in range(n):
				Z,A = self.forward(X_test[i])
				pred = np.argmax(A[-1])
				real = np.argmax(Y_test[i])
				if pred == real:
					correct +=1
			
			print("Epoch: ", epoch, ", Accuracy: ", (correct/n) * 100)


		
		



if __name__=="__main__":
	print(Utils.relu(10))
	print(Utils.relu(-10))
	print(Utils.relu([10,  20, -20.2]))
	
	print(Utils.relu_derivative(np.array([-10, -10, 50])))

	nn = NeuralNetwork([5, 10, 12])
	print(nn.forward(np.ones((5,)) + 0.5)[1])

	print(Utils.softmax(np.array([50,50,50], dtype=np.float128)))