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

    def test_per_class_report_omits_unevaluated_classes(self):
        from types import SimpleNamespace
        from src.evaluate import per_class_metrics
        box = SimpleNamespace(ap_class_index=[7], p=[.8], r=[.5], ap50=[.7], ap=[.4])
        rows = per_class_metrics(box, {0: 'absent', 7: 'bottle'})
        self.assertEqual(1, len(rows))
        self.assertEqual(7, rows[0]['class_id'])
        self.assertEqual(.4, rows[0]['mAP50_95'])
