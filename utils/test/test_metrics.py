"""
# Author = ruben
# Date: 30/8/23
# Project: retinal-framework
# File: test_metrics.py

Description: "Enter description here"
"""
from unittest import TestCase
import math

from utils.metrics import PerformanceMetrics

if __name__ == '__main__':
    pass


def is_float_close(a: float, b: float, rel_tol=1e-09, abs_tol=0.0) -> bool:
    """
    return if tow float are close enough to consider them equal

    :param a: first float
    :param b: second float
    :param rel_tol: relative tolerance, it is multiplied by the greater of the magnitudes of the two arguments;
        as the values get larger, so does the allowed difference between them while still considering them equal
    :param abs_tol: absolute tolerance that is applied as-is in all cases. If the difference is less than either
        of those tolerances, the values are considered equal.
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class TestPerformanceMetrics(TestCase):

    def setUp(self):
        self.valor1 = 2
        self.valor2 = 3
        self.ground = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        self.prediction = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1]
        self.formatted = False
        self.percent = False
        self.performance = PerformanceMetrics(ground=self.ground,
                                              prediction=self.prediction,
                                              percent=self.percent,
                                              formatted=self.formatted)

    def test_accuracy_ok(self):
        accuracy = self.performance.accuracy()
        # type
        self.assertEqual(isinstance(accuracy, float), True,
                         f'Unexpected type for parameter: {type(accuracy)}')
        # value
        self.assertEqual(accuracy, 0.5, f'Wrong computation for parameter')

    def test_precision_ok(self):
        precision = self.performance.precision()
        # type
        self.assertEqual(isinstance(precision, float), True,
                         f'Unexpected type for parameter: {type(precision)}')
        # value
        self.assertAlmostEqual(precision, 0.5, f'Wrong computation for parameter')

    def test_recall_ok(self):
        recall = self.performance.recall()
        # type
        self.assertEqual(isinstance(recall, float), True,
                         f'Unexpected type for parameter: {type(recall)}')
        # value
        self.assertEqual(recall, 0.4, f'Wrong computation for parameter')

    def test_f1_ok(self):
        f1 = self.performance.f1()
        # type
        self.assertEqual(isinstance(f1, float), True,
                         f'Unexpected type for parameter: {type(f1)}')
        # value
        self.assertEqual(is_float_close(f1, 0.44, rel_tol=1e-01), True,
                         f'Wrong computation for parameter')
