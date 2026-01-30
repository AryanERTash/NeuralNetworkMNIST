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

		self.W = [np.ones((layer_info[l], layer_info[l-1])) for l in range(1, len(layer_info))]
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

	



if __name__=="__main__":
	print(Utils.relu(10))
	print(Utils.relu(-10))
	print(Utils.relu([10,  20, -20.2]))
	
	print(Utils.relu_derivative(np.array([-10, -10, 50])))

	nn = NeuralNetwork([5, 10, 12])
	print(nn.forward(np.ones((5,)))[1])