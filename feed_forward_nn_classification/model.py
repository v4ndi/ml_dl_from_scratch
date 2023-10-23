import numpy as np 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
     
class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size, lr):
        """
        input_size - num_features
        hidden_size - hidden_size
        output_size - num_classes
        """
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        # weights for hidden_layer
        self.Wh = np.random.randn(self.input_size, self.hidden_size)
        self.bh = np.zeros((1, self.hidden_size))

        # weights for output_layer
        self.Wo = np.random.randn(self.hidden_size, self.output_size)
        self.bo = np.zeros((1, self.output_size))

        self.epsilon = 1e-10

    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.Wh) + self.bh
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.Wo) + self.bo
        self.output_layer_output = softmax(self.output_layer_input)
        
        return self.output_layer_output
    
    def backward(self, X, y):
        batch_size = X.shape[0]
        d_out = self.output_layer_output - y

        self.Wo -= 1/batch_size * self.lr * np.dot(self.hidden_layer_output.T, d_out)
        self.bo -= 1/batch_size * self.lr * np.sum(d_out, axis=0, keepdims=True)

        d_hid = np.dot(d_out, self.Wo.T) * self.hidden_layer_output * (1 - self.hidden_layer_output)

        self.Wh -= 1/batch_size * self.lr * np.dot(X.T, d_hid)
        self.bh -= 1/batch_size * self.lr * np.sum(d_hid, axis=0, keepdims=True) 

    def __loss(self, y, output):
        output = np.clip(output, self.epsilon, 1 - self.epsilon)

        loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))

        return loss
    
    def predict(self, X):
        output = self.forward(X)
        result = np.zeros_like(output)
        result[np.arange(len(output)), np.argmax(output, axis=1)] = 1

        return result
        
    def train(self, X, y, epochs):
         for epoch in range(epochs):
            output = self.forward(X)

            loss = self.__loss(y, output)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

            self.backward(X, y)