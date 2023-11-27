import numpy as np

class KernelRidgeRegression:
    def __init__(self, alpha=1.0, gamma=1.0):
        self.alpha = alpha  
        self.gamma = gamma  

    def gaussian_kernel(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2)**2)

    def fit(self, X, y):
        self.X = X
        self.y = y
        n_samples, n_features = X.shape

        
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.gaussian_kernel(self.X.iloc[i], self.X.iloc[j])

        
        K += self.alpha * np.identity(n_samples)

        
        self.alpha_coef = np.linalg.solve(K, y)

    def predict(self, X):
        print("X di predict: ", X)
        n_samples, _ = X.shape
        y_pred = np.zeros(n_samples)

        
        for i in range(n_samples):
            k_i = np.array([self.gaussian_kernel(self.X.iloc[i], x) for x in np.array(self.X)])
            y_pred[i] = np.dot(k_i, self.alpha_coef)

        return y_pred
    
    def get_params(self, deep=True):     
        return {'alpha': self.alpha, 'gamma': self.gamma}
