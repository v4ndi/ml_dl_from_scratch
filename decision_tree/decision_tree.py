import numpy as np

class Node():
    def __init__(self, left, right, parent, impurity):
        pass

class DecisionTreeClassifier():
    def __init__(self, num_classes, criterion='gini', max_depth=None, min_samples_split=2):
        assert criterion == 'gini' or criterion == 'entropy', ' Only \"gini\" and \"entropy\" possible criterion'
        assert num_classes >= 2, 'num_classes must be more than 1'
        assert min_samples_split >= 0, 'min_samples_split must me more than zero'

        self.num_classes = num_classes
        self.criterion = 'gini'
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    @staticmethod
    def _entropy(x):
        pass
    
    @staticmethod
    def _gini(x):
        pass
    
    def __impurity(self, x):
        if self.criterion == 'gini':
            return DecisionTreeClassifier._gini(x)
        elif self.criterion == 'entropy':
            return DecisionTreeClassifier._entropy(x)

    def _split_node(self, X, y):
        """Find the best value and feature
        for splitting current Node based on the criterion
        :param X samples in current Node
        :param Y target labels
        """ 
        
        n_features = X.shape[1]
        n_samples = X.shape[0]

        for feature in range(X.shape[0]):
            for val in X[feature]:
                X_left = X[X[feature] <= val]
                X_right =  X[X[feature] > val]

                if X_left and X_right:
                    impurity = (X_left.shape[0] / X.shape[0]) * self.__impurity(X_left) \
                        + (X_left.shape[0] / X.shape[0]) * self.__impurity(X_right)
                    
                    

    # TODO добавить проверку о том, что входные данные это pandas DataFrame или np.array
    def fit(self, X, Y):
        """fit model
        :param X features with shame num_samples, num_features
        :param Y labels from 0 to num_labels - 1
        """
        assert all([isinstance(int, y) for y in Y]), 'The target label must be interger'
        assert min(Y) >= 0 and max(Y) < self.num_classes, 'The target label must be between 0 and num_classes - 1'
        # TODO add test for check shape X and Y
        try:
            X = np.array(X)
            Y = np.array(Y)
        except:
            raise Exception('Coldn\'t convert to numpy array')



