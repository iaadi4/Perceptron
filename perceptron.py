import numpy as np

class Perceptron(object):
    """
    learning_rate (range -> 0 to 1) -- it is a hyperparameter that controls how much to change
    the model in response to estimated error each time the model weight is updated
    
    n_iters -- no of iterations or random trials the search algorithms performs to find the
    best model configuration
    """
    
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        
    def net_input(self, x):
        # dot product of input and weights + bias
        return np.dot(x, self.weights[1:]) + self.weights[0]
    
    def predict(self, x):
        # return 1 if dot product is equal or greater than 0 else -1
        # it is a step function
        return np.where(self.net_input(x) >= 0.0, 1, -1)

    def fit(self, X, y):
        # initialize weights to be zero
        # size will be input size + 1 (for bias)
        self.weights = np.zeros(1 + X.shape[1])
        self.errors = []
        
        print(f"Initial weights: {self.weights}")
        
        # we iterate n_iters time to find best model configuration
        for _ in range(self.n_iters):
            error = 0
            
            # iterating through each input
            for xi, target in zip(X, y):
                
                # calculate the ŷ (predicted value)
                y_pred = self.predict(xi)
                
                # update = learning_rate * (target - ŷ)
                update = self.learning_rate * (target - y_pred)
                
                # update the weights (w = w + Δw)
                self.weights[1:] += update * xi
                
                # update the bias
                self.weights[0] += update
               
                # if update is not equal to zero means target != ŷ 
                error += int(update != 0.0)
            
            # append the error
            self.errors.append(error)
        
        return self