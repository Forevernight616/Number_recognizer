import numpy as np
import random

class Network(object):
    def __init__(self, sizes):
        #sizes -- array of int
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        
    def feedforward(self,a):
        for w,b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w,a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None) -> None:

        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        
    def backprop(self, x, y):
        '''
        x : input vector
        y : desired output vector

        returns : 
        nabla_b[l] = ∂C / ∂b^l
        nabla_w[l] = ∂C / ∂w^l
        the return values are nabla_b and nabla_w, which are the directions to which b and w should
        be nudged to reduce cost function, but b and w are not yet updated here
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        '''
        self.cost_derivative(activations[-1], y) returns output_activations - y
        which is ∂C / ∂a^L = a^L - y, the -1 indicates the last layer L

        sigmoid_prime(zs[-1]) 

        so delta is ∂C / ∂a^L * sigmoid'(z^L) = (a^L - y) * sigmoid'(z^L)
        '''
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # store gradient for output layer
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) 

        '''
        l =1 is the output layer
        l =2 is the last hidden layer
        '''
        for l in range(2, self.num_layers):
            z = zs[-l]
            # sigmoid prime(z^l) tells how sensitive WAS this neuron
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(z)

            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))