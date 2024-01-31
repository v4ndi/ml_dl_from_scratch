from __future__ import annotations
import numpy as np
from typing import Tuple, List, Union
from abc import abstractmethod, ABC
from scipy import stats 


class Node:
    def __init__(self) -> None:
        self.__left = None
        self.__right = None
        self.__feature = None
        self.__split_value = None
        self.leaf_value = None

    def set_params(self, split_value: float, feature: int) -> None:
        self.__split_value = split_value
        self.__feature = feature
    
    def set_children(self, left: Node, right: Node):
        self.__left = left 
        self.__right = right

    def get_params(self) -> Tuple[float, int]:
        return (self.__split_value, self.__feature)
    
    def get_left_node(self) -> Node:
        return self.__left

    def get_right_node(self) -> Node:
        return self.__right


class DecisionTree(ABC):
    def __init__(self, max_depth: int, min_samples_node: int) -> None:
        self.max_depth = max_depth
        self.min_samaples_node = min_samples_node
        self.root = Node()
    
    def __split(self, X: np.array, node: Node, cur_depth: int) -> None:
        allowed_depth = cur_depth <= self.max_depth
        num_classes = np.unique(X[:, -1]).shape[0]

        if allowed_depth and num_classes > 1 and X.shape[0] > self.min_samaples_node:
            best_imp = None
            best_left = None
            best_right = None
            best_feature = None
            best_split_value = None
            
            for feature in range(X.shape[1] - 1):
                for val in np.unique(X[:, feature]):
                    X_left = X[X[:, feature] <= val]
                    X_right =  X[X[:, feature] > val]

                    if X_left.shape[0] > 0 and X_right.shape[0] > 0:
                        impurity = (X_left.shape[0] / X.shape[0]) * self._impurity(X_left) \
                            + (X_right.shape[0] / X.shape[0]) * self._impurity(X_right)
                    
                    if best_imp is None or impurity < best_imp:
                        best_imp = impurity
                        best_left = X_left
                        best_right = X_right
                        best_feature = feature
                        best_split_value = val
            
            node.set_params(split_value=best_split_value, feature=best_feature)
            left, right = Node(), Node()
            node.set_children(left=left, right=right)
            
            self.__split(best_left, left, cur_depth + 1)
            self.__split(best_right, right, cur_depth + 1)
        else:
            node.leaf_value = self._leaf_value(X)
            return

    @abstractmethod
    def _impurity(self):
        pass

    @abstractmethod
    def _leaf_value(self, X):
        pass

    def fit(self, X: np.array, y: np.array) -> None:
        data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        self.__split(X=data, node=self.root, cur_depth=1)
    
    def predict(self, X: np.array) -> np.array:
        precited = []
        for i in range(X.shape[0]):
            cur_node = self.root
            for _ in range(self.max_depth):
                if cur_node.leaf_value is None:
                    split_value, feature = cur_node.get_params()
                    if X[i, feature] > split_value:
                        cur_node = cur_node.get_right_node()
                    else:
                        cur_node = cur_node.get_left_node()
                else:
                    break

            precited.append(cur_node.leaf_value)
        return np.array(precited)


class DecisionTreeClassifier(DecisionTree):
    def __init__(self, max_depth: int, min_samples_node: int, impurity='entropy') -> None:
        super().__init__(max_depth, min_samples_node)
        assert impurity == 'entropy' or impurity == 'gini', "available loss is gini or entropy"
        self.impurity = impurity

    def __gini(self, X: np.array) -> float:
        gini = 0
        for cls in np.unique(X[:, -1]):
            p = X[X[:, -1] == cls].shape[0] / X.shape[0]
            gini += p * (1 - p)
        
        return gini

    def __entropy(self, X: np.array) -> float:
        entropy = 0
        for cls in np.unique(X[:, -1]):
            p = X[X[:, -1] == cls].shape[0] / X.shape[0]
            entropy -= p * np.log2(p)
        
        return entropy

    def _impurity(self, X: np.array):
        if self.impurity == 'gini':
            return self.__gini(X)
        else:
            return self.__entropy(X)

    def _leaf_value(self, X) -> int:
        return stats.mode(X[:, -1])[0]
    

class DecisionTreeRegressor(DecisionTree):
    def __init__(self, max_depth: int, min_samples_node: int, loss='mse') -> None:
        """
        params:
        max_depeth: int max depth of tree
        min_samples_node: int min samples of node
        loss: str 'mse' or'mae' loss of regression
        """
        super().__init__(max_depth=max_depth, min_samples_node=min_samples_node)
        self.loss = loss

    def _impurity(self, X: np.array) -> float:
        if self.loss == 'mse':
            return self.__mse(X)
        else:
            return self.__mae(X)

    def __mse(self, X: np.array) -> float:
        return np.mean((X[:, -1] - np.mean(X[:, -1]))**2)
    
    def __mae(self, X: np.array) -> float:
        return np.mean(np.abs(X[:, - 1] - np.mean(X[:, -1])))
    
    def _leaf_value(self, X: np.array) -> float:
        return np.mean(X[:, -1])