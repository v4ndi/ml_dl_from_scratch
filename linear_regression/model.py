import numpy as np

class LinearRegression():
    def __init__(self):
        self.w = None

    def fit(self, X, y, lr=0.001, n_iterations=1000, intercept=True, verbose=10):
        '''
        X.shape -> [n_samples, n_features]
        y.shape -> [n_samples]
        '''
        self.n_samples, self.n_features = X.shape
        self.X = X 
        self.y = y
        self.intercept = intercept
        
        if self.intercept:
            self.w = np.random.randn(self.n_features + 1)
            self.X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        else:
            self.w = np.random.randn(self.n_features)

        for i in range(n_iterations):
            outputs = np.dot(self.X, self.w)
            
            loss = self.loss(self.y, outputs)
            self.w -= lr * self.get_grad(self.X, self.y, outputs)

            if i % verbose == 0:
                print(loss)

    def predict(self, X):
        n_samples, n_features = X.shape
        assert self.n_features == n_features or self.n_samples == n_samples, 'Training data had another shape'
        
        if self.intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        
        return np.dot(X, self.w)

    def get_weights(self):
        cp_w = np.copy(self.w)
        return cp_w

    def loss(self, y, y_pred):
        # MSE
        return np.mean((y_pred - y)**2)

    def get_grad(self, X, y, y_pred):
        N = X.shape[0]
        dL_dw = 2/len(X) * np.dot(X.T, (y_pred - y)) 

        return dL_dw