import numpy as np

class Perceptron(object):
    
    """
    learning rate (range -> 0 to 1) -- it is a hyperparameter that controls how much to change
    the model in response to estimated error each time the model weight is updated
    
    n_iters -- no of iterations or random trials the search algorithms performs to find the
    best model configuration
    """
    
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        
    def weighted_sum(self, x):
        # dot product of input and weights + bias
        return np.dot(self.weighted_sum[1:], x) + self.weighted_sum[0]
    
    def predict(self, x):
        # return 1 if dot product is equal or greater than 0 else -1
        # it is a step function
        return np.where(self.weighted_sum(self, x) >= 0.0, 1, -1)

    def fit(self, x, y):
        
        # initialize weights to be zero
        # size will be input size + 1 (for bias)
        self.weighted_sum = np.zeros(1 + x.shape[1])
        self.errors = []
        
        print(f"Weighted_sum: {weighted_sum}")
        
        # we iterate n_iters time to find best model configuration
        for _ in range(n_iters):
            error = 0
            
            # iterating through each input
            # xi is the ith input
            for xi, y in zip(x, y):
                
                # calculate the ŷ (predicted value)
                y_pred = self.predict(xi)
                
                # update = learning_rate * (y - ŷ)
                update = self.learning_rate * (y - y_pred)
                
                # update the weights (w = w + Δw)
                self.weighted_sum[1:] = self.weighted_sum[1:] + update * xi
                print(f"updated weight: {self.weighted_sum}")
                
                # update the bias
                self.weighted_sum[0] = self.weighted_sum[0] + update
               
                # if update is not equal to zero means y  != ŷ (output != predicted value)
                error += int(update != 0)
            
            # append the error
            self.errors.append(error);
        
        return self
