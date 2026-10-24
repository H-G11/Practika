import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src import detect, train
from src.realtime import run_camera


class ExecutionTests(unittest.TestCase):
    def test_detect_uses_positional_source_and_explicit_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, weights = root / 'test.jpg', root / 'best.pt'
            source.touch()
            weights.touch()
            factory = Mock()
            with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=factory)}):
                detect.main([str(source), '--weights', str(weights)])
            self.assertEqual(str(source.resolve()), factory.return_value.predict.call_args.kwargs['source'])
            self.assertEqual('cpu', factory.return_value.predict.call_args.kwargs['device'])

    def test_missing_source_fails_without_loading_model(self):
        with self.assertRaises(SystemExit) as exc:
            detect.main(['missing.jpg', '--weights', 'missing.pt'])
        self.assertEqual(2, exc.exception.code)

    def test_training_is_cpu_by_default_and_parameters_are_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            config, weights = Path(directory) / 'data.yaml', Path(directory) / 'model.pt'
            config.touch()
            weights.touch()
            factory = Mock()
            with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=factory)}):
                train.main(['--data', str(config), '--weights', str(weights), '--epochs', '2', '--seed', '17'])
            args = factory.return_value.train.call_args.kwargs
            self.assertEqual(('cpu', 2, 17), (args['device'], args['epochs'], args['seed']))

    def test_camera_released_after_predict_failure(self):
        cv2, model = Mock(), Mock()
        cv2.VideoCapture.return_value.read.return_value = (True, object())
        model.predict.side_effect = RuntimeError('synthetic failure')
        with self.assertRaises(RuntimeError):
            run_camera(model, cv2, camera=1, conf=.4, imgsz=640, device='cpu')
        cv2.VideoCapture.assert_called_once_with(1)
        cv2.VideoCapture.return_value.release.assert_called_once()
        cv2.destroyAllWindows.assert_called_once()

    def test_camera_read_failure_is_reported_and_releases_resources(self):
        cv2 = Mock()
        cv2.VideoCapture.return_value.read.return_value = (False, None)
        with self.assertRaisesRegex(RuntimeError, 'delivering frames'):
            run_camera(Mock(), cv2, camera=0, conf=.4, imgsz=640, device='cpu')
        cv2.VideoCapture.return_value.release.assert_called_once()

    def test_quit_releases_camera(self):
        cv2, model = Mock(), Mock()
        cv2.VideoCapture.return_value.read.return_value = (True, object())
        cv2.waitKey.return_value = ord('q')
        model.predict.return_value = [Mock()]
        run_camera(model, cv2, camera=0, conf=.4, imgsz=640, device='cpu')
        cv2.VideoCapture.return_value.release.assert_called_once()
        self.assertEqual(1, model.predict.call_count)
