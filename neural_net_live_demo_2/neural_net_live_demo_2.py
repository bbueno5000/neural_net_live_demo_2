"""
Simplistic implementation of the two-layer neural network.
Training method is stochastic (online) gradient descent with momentum.

As an example it computes XOR for given input.
"""
import numpy
import time

learning_rate = 0.01
momentum = 0.9
n_hidden = 10
n_in = 10
n_out = 10
n_samples = 300

numpy.random.seed(0)

class NeuralNet:
    """
    Details:
    - tanh activation for hidden layer
    - sigmoid activation for output layer
    - cross-entropy loss
    """
    def predict(self, x, V, W, bv, bw):
        """
        DOCSTRING
        """
        A = numpy.dot(x, V) + bv
        B = numpy.dot(numpy.tanh(A), W) + bw
        return (self.sigmoid(B) > 0.5).astype(int)
    
    def sigmoid(self, x):
        """
        DOCSTRING
        """
        return 1.0 / (1.0 + numpy.exp(-x))

    def tanh_prime(self, x):
        """
        DOCSTRING
        """
        return 1 - numpy.tanh(x)**2

    def train(self, x, t, V, W, bv, bw):
        """
        DOCSTRING
        """
        # forward
        A = numpy.dot(x, V) + bv
        Z = numpy.tanh(A)
        B = numpy.dot(Z, W) + bw
        Y = self.sigmoid(B)
        # backward
        Ew = Y - t
        Ev = self.tanh_prime(A) * numpy.dot(W, Ew)
        dW = numpy.outer(Z, Ew)
        dV = numpy.outer(x, Ev)
        loss = -numpy.mean (t * numpy.log(Y) + (1 - t) * numpy.log(1 - Y))
        # NOTE: we use error for each layer as a gradient for biases
        return  loss, (dV, dW, Ev, Ew)

if __name__ == '__main__':
    # initialize parameters
    V = numpy.random.normal(scale=0.1, size=(n_in, n_hidden))
    W = numpy.random.normal(scale=0.1, size=(n_hidden, n_out))
    bv = numpy.zeros(n_hidden)
    bw = numpy.zeros(n_out)
    params = [V, W, bv, bw]
    # generate data
    X = numpy.random.binomial(1, 0.5, (n_samples, n_in))
    T = X ^ 1
    # train
    for epoch in range(100):
        err, upd = list(), [0] * len(params)
        t0 = time.clock()
        for i in range(X.shape[0]):
            loss, grad = NeuralNet.train(X[i], T[i], *params)
            for j in range(len(params)):
                params[j] -= upd[j]
            for j in range(len(params)):
                upd[j] = learning_rate * grad[j] + momentum * upd[j]
            err.append(loss)
            print('Epoch: {d}, Loss: {0.8f}, Time: {0.4f}s'.format(
                epoch, numpy.mean(err), time.clock()-t0))
    # make prediction
    x = numpy.random.binomial(1, 0.5, n_in)
    print('XOR prediction:')
    print(x)
    print(NeuralNet.predict(x, *params))
