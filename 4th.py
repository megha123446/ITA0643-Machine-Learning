import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.biases_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.biases_output = np.zeros((1, output_size))

    def forward(self, inputs):
        self.hidden = sigmoid(np.dot(inputs, self.weights_input_hidden) + self.biases_hidden)
        self.output = sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.biases_output)
        return self.output

    def backward(self, inputs, targets, learning_rate):
        output_error = targets - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

        self.weights_hidden_output += self.hidden.T.dot(output_delta) * learning_rate
        self.biases_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate
        self.biases_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        for _ in range(epochs):
            for i in range(len(inputs)):
                self.forward(inputs[i:i+1])
                self.backward(inputs[i:i+1], targets[i:i+1], learning_rate)

    def predict(self, inputs):
        return self.forward(inputs)

# XOR Dataset
inputs = np.array([[0, 0], 
                   [0, 1], 
                   [1, 0], 
                   [1, 1]])

targets = np.array([[0], 
                    [1], 
                    [1], 
                    [0]])

# Create and train the neural network
neural_network = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
neural_network.train(inputs, targets, epochs=10000, learning_rate=0.1)

# Test the neural network
for i in range(len(inputs)):
    prediction = neural_network.predict(inputs[i:i+1])
    print(f"Input: {inputs[i]}, Target: {targets[i]}, Prediction: {prediction}")