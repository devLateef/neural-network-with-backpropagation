import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = [] # initialize the list of weights matrices
        self.layers = layers # store the network architecture
        self.alpha = alpha # store learning rate
        for i in np.arange(0, len(layers) - 2) # Loop from first layer index then stop before reaching last 2 layers
        w = np.random.randn(layers[i] + 1, layers[i + 1] + 1) # Randomly initialize weight, add extra node for the bias
        self.W.append(w / np.sqrt(layers[i])) # Normalize it

        