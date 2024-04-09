"""
# Author: ruben 
# Date: 30/8/23
# Project: retinal-framework
# File: metrics.py

Description: Functions to provide performance metrics
"""
import statistics
import math
from sklearn.metrics import multilabel_confusion_matrix

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score, accuracy_score

# Formatting number of decimals
ND = 2


def hamming_score(ground: np.array, prediction: np.array,):
    """
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    :param ground: input array of ground truth
    :param prediction: input array of prediction values

    http://stackoverflow.com/q/32239577/395857
    """
    acc_list = []
    for i in range(ground.shape[0]):
        set_ground = set(np.where(ground[i])[0])
        set_prediction = set(np.where(prediction[i])[0])
        if len(set_ground) == 0 and len(set_prediction) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_ground.intersection(set_prediction)) / \
                    float(len(set_ground.union(set_prediction)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def dice_loss(im1: np.array, im2: np.array):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


class CrossValidationMeasures:
    """
    Class to get cross validation model performance of all folds
    """

    def __init__(self, measures_list, confidence=1.96, percent=False, formatted=False):
        """
        CrossValidationMeasures constructor
        :param measures_list: (list) List of measures by fold
        :param confidence: (float) Confidence interval percentage
        :param percent: (bool) whether if data is provided [0-1] or [0%-100%]
        :param formatted: (bool) whether if data is formatted with 2 decimals or not.
            If activated return string instead of float
        """
        assert (len(measures_list) > 0)
        self._measures = measures_list
        self._confidence = confidence
        self._percent = percent
        self._formatted = formatted
        self._compute_performance()

    def _compute_performance(self):
        """
        Compute mean, std dev and CI of a list of model measures
        """

        if len(self._measures) == 1:
            self._mean = self._measures[0]
            self._stddev = 0.0
            self._offset = 0.0
        else:
            self._mean = statistics.mean(self._measures)
            self._stddev = statistics.stdev(self._measures)
            self._offset = self._confidence * self._stddev / math.sqrt(len(self._measures))
        self._interval = self._mean - self._offset, self._mean + self._offset

    def mean(self):
        """Compute Mean value"""
        if self._percent and self._measures[0] <= 1.0:
            mean = self._mean * 100.0
        else:
            mean = self._mean
        if self._formatted:
            return f'{mean:.{ND}f}'
        else:
            return mean

    def stddev(self):
        """Compute Std Dev value"""
        if self._percent and self._measures[0] <= 1.0:
            stddev = self._stddev * 100.0
        else:
            stddev = self._stddev
        if self._formatted:
            return f'{stddev:.{ND}f}'
        else:
            return stddev

    def interval(self):
        """Compute Confidence interval"""
        if self._percent:
            interval = self._interval[0] * 100.0, self._interval[1] * 100.0
        else:
            interval = self._interval[0], self._interval[1]
        if self._formatted:
            return f'({interval[0]:.{ND}f}, {interval[1]:.{ND}f})'
        else:
            return interval


class PerformanceMetrics:
    """
    Class to compute model performance
    """

    def __init__(self, ground, prediction, percent=False, formatted=False):
        """
        PerformanceMetrics class constructor
        :param ground: input array of ground truth
        :param prediction: input array of prediction values
        :param percent: (bool) whether if data is provided [0-1] or [0%-100%]
        :param formatted: (bool) whether if data is formatted with 2 decimals or not.
            If activated return string instead of float
        """
        assert (len(ground) == len(prediction))
        self._ground = ground
        self._prediction = prediction
        self._percent = percent
        self._formatted = formatted
        self._confusion_matrix = None
        self._accuracy = None
        self._precision = None
        self._recall = None
        self._f1 = None
        self._compute_measures()

    def _compute_measures(self):
        """
        Compute performance measures
        """
        self._compute_confusion_matrix()
        self._compute_accuracy()
        self._compute_precision()
        self._compute_recall()
        self._compute_f1()

    def _compute_confusion_matrix(self):
        """
        Computes the confusion matrix of a model
        """
        self._tp, self._fp, self._tn, self._fn = 0, 0, 0, 0

        for i in range(len(self._prediction)):
            if self._ground[i] == self._prediction[i] == 1:
                self._tp += 1
            if self._prediction[i] == 1 and self._ground[i] != self._prediction[i]:
                self._fp += 1
            if self._ground[i] == self._prediction[i] == 0:
                self._tn += 1
            if self._prediction[i] == 0 and self._ground[i] != self._prediction[i]:
                self._fn += 1

        self._confusion_matrix = self._tn, self._fp, self._fn, self._tp

    def _compute_accuracy(self):
        """Computes the accuracy of a model"""
        self._accuracy = (self._tn + self._tp) / len(self._prediction)

    def _compute_precision(self):
        """Computes the precision of a model"""
        try:
            self._precision = self._tp / (self._tp + self._fp)
        except ZeroDivisionError:
            self._precision = 0.0

    def _compute_recall(self):
        """Computes the recall of a model"""
        try:
            self._recall = self._tp / (self._tp + self._fn)
        except ZeroDivisionError:
            self._recall = 0.0

    def _compute_f1(self):
        """Computes the F1 measure of a model"""
        try:
            self._f1 = 2 * (self._precision * self._recall / (self._precision + self._recall))
        except ZeroDivisionError:
            self._f1 = 0.0

    def confusion_matrix(self):
        """Return Confusion matrix"""
        return self._confusion_matrix

    def accuracy(self):
        """Return Accuracy measure"""
        if self._percent:
            accuracy = self._accuracy * 100.0
        else:
            accuracy = self._accuracy
        if self._formatted:
            return f'{accuracy:.{ND}f}'
        else:
            return accuracy

    def precision(self):
        """Return Precision measure"""
        if self._percent:
            precision = self._precision * 100.0
        else:
            precision = self._precision
        if self._formatted:
            return f'{precision:.{ND}f}'
        else:
            return precision

    def recall(self):
        """Return Recall measure"""
        if self._percent:
            recall = self._recall * 100.0
        else:
            recall = self._recall
        if self._formatted:
            return f'{recall:.{ND}f}'
        else:
            return recall

    def f1(self):
        """Return F1 measure"""
        if self._percent:
            f1 = self._f1 * 100.0
        else:
            f1 = self._f1
        if self._formatted:
            return f'{f1:.{ND}f}'
        else:
            return f1


class PerformanceMetricsMultiClassSKL:
    """
    Class to compute model performance for multi-class classification using scikit-learn lib
    """

    def __init__(self, ground: list, prediction: list, classes: dict, percent=False, formatted=False):
        """
        Class constructor
        :param ground: input array of ground truth
        :param prediction: input array of prediction values
        :param percent: (bool) whether if data is provided [0-1] or [0%-100%]
        :param formatted: (bool) whether if data is formatted with 2 decimals or not.
            If activated return string instead of float
        """
        assert (len(ground) == len(prediction))
        self._ground = ground
        self._prediction = prediction
        self._percent = percent
        self._formatted = formatted
        self._classes = classes
        self._confusion_matrix = dict.fromkeys(classes)
        self._accuracy = dict.fromkeys(classes)
        self._precision = dict.fromkeys(classes)
        self._recall = dict.fromkeys(classes)
        self._f1 = dict.fromkeys(classes)
        self._compute_measures()

    def _compute_measures(self):
        """
        Compute performance measures
        """
        cm = multilabel_confusion_matrix(np.array(self._ground), np.array(self._prediction))
        self._tn = cm[:, 0, 0]
        self._tp = cm[:, 1, 1]
        self._fn = cm[:, 1, 0]
        self._fp = cm[:, 0, 1]

        for cls in self._classes:
            self._confusion_matrix[cls] = [self._tn[cls], self._tp[cls], self._fn[cls], self._fp[cls]]

        # self._accuracy = (self._tp + self._tn) / (self._tp + self._fn + self._tn + self._fp)
        self._accuracy = float(accuracy_score(np.array(self._ground), np.array(self._prediction)))
        self._precision = self._tp / (self._tp + self._fp)
        self._recall = self._tp / (self._tp + self._fn)
        self._f1 = 2 * (self._precision * self._recall) / (self._precision + self._recall)

    def confusion_matrix(self) -> np.array:
        """
        :return: Confusion matrix
        """
        return self._confusion_matrix

    def accuracy(self):
        """Return Accuracy measure"""
        accuracy = None
        if self._percent:
            accuracy = self._accuracy * 100.0
        else:
            accuracy = self._accuracy

        if self._formatted:
            accuracy = f'{accuracy:.{ND}f}'

        return accuracy

    def precision(self):
        """Return Precision measure"""
        precision = dict.fromkeys(self._classes)
        for cls in self._classes:
            if self._percent:
                precision[cls] = self._precision[cls] * 100.0
            else:
                precision[cls] = self._precision[cls]
            if self._formatted:
                precision[cls] = f'{precision[cls]:.{ND}f}'

        return precision

    def recall(self):
        """Return Recall measure"""
        recall = dict.fromkeys(self._classes)
        for cls in self._classes:
            if self._percent:
                recall[cls] = self._recall[cls] * 100.0
            else:
                recall[cls] = self._recall[cls]
            if self._formatted:
                recall[cls] = f'{recall[cls]:.{ND}f}'
        return recall

    def f1(self):
        """Return F1 measure"""
        f1 = dict.fromkeys(self._classes)
        for cls in self._classes:
            if self._percent:
                f1[cls] = self._f1[cls] * 100.0
            else:
                f1[cls] = self._f1[cls]
            if self._formatted:
                f1[cls] = f'{f1[cls]:.{ND}f}'
        return f1


class PerformanceMetricsMultiClass:
    """
    Class to compute model performance for multi-class classification, custom implementation
    """

    def __init__(self, ground: list, prediction: list, classes: dict, percent=False, formatted=False):
        """
        Class constructor
        :param ground: input array of ground truth
        :param prediction: input array of prediction values
        :param percent: (bool) whether if data is provided [0-1] or [0%-100%]
        :param formatted: (bool) whether if data is formatted with 2 decimals or not.
            If activated return string instead of float
        """
        assert (len(ground) == len(prediction))
        self._ground = ground
        self._prediction = prediction
        self._percent = percent
        self._formatted = formatted
        self._classes = classes
        self._confusion_matrix = dict.fromkeys(classes)
        self._accuracy = dict.fromkeys(classes)
        self._precision = dict.fromkeys(classes)
        self._recall = dict.fromkeys(classes)
        self._f1 = dict.fromkeys(classes)
        self._compute_measures()

    def _compute_measures(self):
        """Compute performance measures"""
        for cls in self._classes:
            self._compute_confusion_matrix(cls)
            self._compute_accuracy(cls)
            self._compute_precision(cls)
            self._compute_recall(cls)
            self._compute_f1(cls)

    def _compute_confusion_matrix(self, cls):
        """Computes the confusion matrix of a model"""
        self._tp, self._fp, self._tn, self._fn = 0, 0, 0, 0
        ground = []
        prediction = []
        if cls == 0:
            for i in range(len(self._prediction)):
                if self._prediction[i] != cls:
                    prediction.append(1)
                else:
                    prediction.append(cls)
                if self._ground[i] != cls:
                    ground.append(1)
                else:
                    ground.append(cls)
        elif cls == 1:
            for i in range(len(self._prediction)):
                if self._prediction[i] != cls:
                    prediction.append(0)
                else:
                    prediction.append(cls)
                if self._ground[i] != cls:
                    ground.append(0)
                else:
                    ground.append(cls)
        elif cls == 2:
            for i in range(len(self._prediction)):
                if self._prediction[i] != cls:
                    prediction.append(0)
                else:
                    prediction.append(1)
                if self._ground[i] != cls:
                    ground.append(0)
                else:
                    ground.append(1)

        for i in range(len(self._prediction)):
            if ground[i] == prediction[i] == 1:
                self._tp += 1
            if prediction[i] == 1 and ground[i] != prediction[i]:
                self._fp += 1
            if ground[i] == prediction[i] == 0:
                self._tn += 1
            if prediction[i] == 0 and ground[i] != prediction[i]:
                self._fn += 1
        self._confusion_matrix[cls] = self._tn, self._fp, self._fn, self._tp

    def _compute_accuracy(self, cls: str):
        """Computes the accuracy of a model"""
        self._accuracy[cls] = (self._tn + self._tp) / len(self._prediction)

    def _compute_precision(self, cls: str):
        """Computes the precision of a model"""
        try:
            self._precision[cls] = self._tp / (self._tp + self._fp)
        except ZeroDivisionError:
            self._precision[cls] = 0.0

    def _compute_recall(self, cls: str):
        """Computes the recall of a model"""
        try:
            self._recall[cls] = self._tp / (self._tp + self._fn)
        except ZeroDivisionError:
            self._recall[cls] = 0.0

    def _compute_f1(self, cls: str):
        """Computes the F1 measure of a model"""
        try:
            self._f1[cls] = 2 * (self._precision[cls] * self._recall[cls] / (self._precision[cls] + self._recall[cls]))
        except ZeroDivisionError:
            self._f1[cls] = 0.0

    def confusion_matrix(self) -> np.array:
        """Return Confusion matrix"""
        return self._confusion_matrix

    def accuracy(self):
        """Return Accuracy measure"""
        accuracy = dict.fromkeys(self._classes)
        for cls in self._classes:
            if self._percent:
                accuracy[cls] = self._accuracy[cls] * 100.0
            else:
                accuracy[cls] = self._accuracy[cls]

            if self._formatted:
                accuracy[cls] = f'{accuracy[cls]:.{ND}f}'

        return accuracy

    def precision(self):
        """Return Precision measure"""
        precision = dict.fromkeys(self._classes)
        for cls in self._classes:
            if self._percent:
                precision[cls] = self._precision[cls] * 100.0
            else:
                precision[cls] = self._precision[cls]
            if self._formatted:
                precision[cls] = f'{precision[cls]:.{ND}f}'

        return precision

    def recall(self):
        """Return Recall measure"""
        recall = dict.fromkeys(self._classes)
        for cls in self._classes:
            if self._percent:
                recall[cls] = self._recall[cls] * 100.0
            else:
                recall[cls] = self._recall[cls]
            if self._formatted:
                recall[cls] = f'{recall[cls]:.{ND}f}'
        return recall

    def f1(self):
        """Return F1 measure"""
        f1 = dict.fromkeys(self._classes)
        for cls in self._classes:
            if self._percent:
                f1[cls] = self._f1[cls] * 100.0
            else:
                f1[cls] = self._f1[cls]
            if self._formatted:
                f1[cls] = f'{f1[cls]:.{ND}f}'
        return f1
