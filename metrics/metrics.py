import numpy as np
from typing import Union, Literal


def mse(y_true: np.array, y_pred: np.array) -> float:
    return np.mean((y_true - y_pred)**2)


def mae(y_true: np.array, y_pred: np.array) -> float:
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.array, y_pred: np.array) -> float:
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true: np.array, y_pred: np.array) -> float:
    mean_true = np.mean(y_true)
    ss_total = np.sum(((y_true - mean_true) ** 2))
    ss_resudial = np.sum(((y_true - y_pred) ** 2))
    return 1 - ss_resudial / ss_total


def confussion_matrix(y_true: np.array, y_pred: np.array) -> np.array:
    """
    Assume that vertical axis is true label and horizoontal axis is predicted label
    """
    num_classes = np.unique(np.concatenate((y_true, y_pred))).shape[0]
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(num_classes):
        for j in range(num_classes):
            matrix[i, j] = np.sum(np.logical_and(y_true == i, y_pred == j))

    return matrix


def roc_curve(y_true: np.array, y_pred_prob: np.array):
    def TPR(y_true: np.array, y_pred_prob: np.array, threshold: float) -> float:
        """
        True possitive rate for binary classification (TPR)
        TPR = TP / (TP + FN)
        """
        y_pred = y_pred_prob >= threshold
        tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
        fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))
        return tp / (tp + fn)


    def FPR(y_true: np.array, y_pred_prob: np.array, threshold: float) -> float:
        """
        Faile possitive rate for binary classification (FPR)
        FPR = FP / (FP + TN)
        """
        y_pred = y_pred_prob >= threshold
        fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))
        tn = np.sum(np.logical_and(y_true == 0, y_pred == 0))
        return fp / (fp + tn)
    
    tpr_list = [0]
    fpr_list = [0]
    sorted_indexes = np.argsort(y_pred_prob)[::-1]
    y_true = y_true[sorted_indexes]
    y_pred_prob = y_pred_prob[sorted_indexes]
    for threshold in y_pred_prob:
        tpr_list.append(TPR(y_true, y_pred_prob, threshold))
        fpr_list.append(FPR(y_true, y_pred_prob, threshold))
    
    fpr_list.append(1)
    tpr_list.append(1)
    return fpr_list, tpr_list


def roc_auc_score(y_true: np.array, y_pred_prob: np.array) -> float:
    fpr, tpr = roc_curve(y_true, y_pred_prob)
    auc = 0.

    for i in range(1, len(tpr)):
        a = 1 - fpr[i]
        b = 1 - fpr[i - 1]
        h = tpr[i] - tpr[i - 1]
        auc += (a + b) / 2 * h

    return auc


def accuracy_score(y_true: np.array, y_pred: np.array):
    tp = np.sum(y_true == y_pred)
    total = y_true.shape[0]
    return tp / total


def precision_score(y_true: np.array,
                    y_pred: np.array,
                    average: Union[Literal['binary'],Literal['micro'], Literal['macro']] = 'binary'
                    ) -> float:
    if average == 'binary':
        tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
        fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))
        
        return  tp / (tp + fp)
    elif average == 'micro':
        num = 0. # numerator
        den = 0. # denominator 
        num_classes = np.unique(np.concatenate((y_true, y_pred))).shape[0]

        for i in range(num_classes):
            tp_i = np.sum(np.logical_and(y_true == i, y_pred == i))
            fp_i = np.sum(np.logical_and(y_true != i, y_pred == i))
            num += tp_i
            den += fp_i + tp_i

        return num / den
    elif average == 'macro':
        precision_list = []
        num_classes = np.unique(np.concatenate((y_true, y_pred))).shape[0]

        for i in range(num_classes):
            tp_i = np.sum(np.logical_and(y_true == i, y_pred == i))
            fp_i = np.sum(np.logical_and(y_true != i, y_pred == i))
            precision_list.append(tp_i / (tp_i + fp_i))
        
        return np.mean(precision_list)
    
    raise ValueError(f'average {average} is not supported')


def recall_score(y_true: np.array,
                 y_pred: np.array,
                 average: Union[Literal['binary'],Literal['micro'], Literal['macro']] = 'binary'
                 ) -> float:
    if average == 'binary':
        tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
        fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))
        
        return  tp / (tp + fn)
    elif average == 'micro':
        num = 0. # numerator
        den = 0. # denominator 
        num_classes = np.unique(np.concatenate((y_true, y_pred))).shape[0]

        for i in range(num_classes):
            tp_i = np.sum(np.logical_and(y_true == i, y_pred == i))
            fn_i = np.sum(np.logical_and(y_true == i, y_pred != i))
            num += tp_i
            den += fn_i + tp_i

        return num / den
    elif average == 'macro':
        recall_list = []
        num_classes = np.unique(np.concatenate((y_true, y_pred))).shape[0]

        for i in range(num_classes):
            tp_i = np.sum(np.logical_and(y_true == i, y_pred == i))
            fn_i = np.sum(np.logical_and(y_true == i, y_pred != i))
            recall_list.append(tp_i / (tp_i + fn_i))
        
        return np.mean(recall_list)
    
    raise ValueError(f'average {average} is not supported')


def f1_score(y_true: np.array,
            y_pred: np.array,
            average: Union[Literal['binary'],Literal['micro'], Literal['macro']] = 'binary'
            ) -> float:
    if average == 'binary':
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall)
    elif average == 'micro':
        precision_micro = precision_score(y_true, y_pred, average='micro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        return 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)
    elif average == 'macro':
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        return 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro)

    raise ValueError(f'average {average} is not supported')


def fbeta_score(y_true,
                y_pred,
                beta=1.0,
                average='binary'
                ) -> float:
    assert beta >= 0, 'betas must be non-negative'

    if average == 'binary':
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        return  (2 * (precision * recall) / (beta * precision + recall)) * (1 + beta**2)
    elif average == 'micro':
        precision_micro = precision_score(y_true, y_pred, average='micro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        return (2 * (precision_micro * recall_micro) / (beta * precision_micro + recall_micro)) * (1 + beta**2)
    elif average == 'macro':
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        return (2 * (precision_macro * recall_macro) / (beta * precision_macro + recall_macro)) * (1 + beta**2)

    raise ValueError(f'average {average} is not supported')

#TODO 
#Compute the average Hamming loss.
#Average hinge loss (non-regularized).
#Jaccard similarity coefficient score.
#Log loss, aka logistic loss or cross-entropy loss.