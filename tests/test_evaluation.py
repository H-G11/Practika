import unittest

from src.evaluate import confusion_counts


class ConfusionTests(unittest.TestCase):
    def test_cross_class_confusion_counts_as_fp_and_fn(self):
        # 2 foreground classes and one background row/column.
        counts = confusion_counts([[5, 2, 1], [3, 7, 4], [6, 8, 0]])
        self.assertEqual({'true_positives': 12., 'false_positives': 10., 'false_negatives': 19.}, counts)

    def test_invalid_matrix_rejected(self):
        for matrix in ([], [[1]], [[1, 2], [3]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                confusion_counts(matrix)
