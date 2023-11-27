import numpy as np
class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  

    def fit(self, X, y):
      
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
       
        n_features = X.shape[1]
        identity_matrix = np.eye(n_features)
        self.coefficients = np.linalg.inv(X.T.dot(X) + self.alpha * identity_matrix).dot(X.T).dot(y)

    def predict(self, X):
        
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)

       
        return X.dot(self.coefficients)


    def get_params(self, deep=True):
        
        return {'alpha': self.alpha}