import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.prediction import detection_report, predict_image
from src.viewer import predict_upload


class PredictionTests(unittest.TestCase):
    def test_only_local_images_and_finite_thresholds_reach_model(self):
        model = Mock()
        with self.assertRaises(ValueError):
            predict_image(model, 'https://example.invalid/image.jpg')
        model.predict.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'image.jpg'
            source.touch()
            for options in ({'confidence': float('nan')}, {'iou': float('inf')}, {'iou': 0}, {'imgsz': 0}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    predict_image(model, source, **options)
            model.predict.assert_not_called()
            model.predict.return_value = ['actual-result-object']
            self.assertEqual('actual-result-object', predict_image(model, source, confidence=.4, iou=.5))
            self.assertEqual(.5, model.predict.call_args.kwargs['iou'])
            self.assertEqual(str(source.resolve()), model.predict.call_args.kwargs['source'])

    def test_empty_detections_do_not_create_a_fake_box(self):
        result = SimpleNamespace(boxes=None, orig_shape=(20, 40), speed={'inference': 1.2}, names={0: 'trash'})
        self.assertEqual([], detection_report(result)['detections'])
        self.assertEqual(0, detection_report(result)['count'])

    def test_export_uses_actual_class_confidence_and_coordinates(self):
        boxes = SimpleNamespace(xyxy=Mock(), cls=Mock(), conf=Mock())
        boxes.xyxy.tolist.return_value = [[2, 3, 12, 19]]
        boxes.cls.tolist.return_value = [7]
        boxes.conf.tolist.return_value = [.83]
        result = SimpleNamespace(boxes=boxes, names={7: 'bottle'}, orig_shape=(20, 40), speed={})
        row = detection_report(result)['detections'][0]
        self.assertEqual({'class_id': 7, 'label': 'bottle', 'confidence': .83, 'xyxy': [2., 3., 12., 19.]}, row)
