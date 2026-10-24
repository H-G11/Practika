import json
import tempfile
import unittest
from pathlib import Path

from src.coco_to_yolo import convert, image_path, yolo_bbox


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        self.data = {'categories': [{'id': 7, 'name': 'foil: "metal"'}], 'images': [], 'annotations': []}
        for i in range(10):
            (self.images / f'{i}.jpg').write_bytes(f'fixture {i}'.encode())
            self.data['images'].append({'id': i, 'file_name': f'{i}.jpg', 'width': 100, 'height': 80})
            if i:  # negative images must stay in the dataset
                self.data['annotations'].append({'id': i, 'image_id': i, 'category_id': 7, 'bbox': [10, 20, 20, 40]})
        self.annotations = self.root / 'annotations.json'
        self.output = self.root / 'output'

    def convert(self, output=None):
        self.annotations.write_text(json.dumps(self.data))
        return convert(self.annotations, self.images, output or self.output)

    def test_reproducible_disjoint_splits_and_real_boxes(self):
        a = self.convert()
        self.data['images'].reverse()
        b = self.convert(self.root / 'another')
        self.assertEqual(a['splits'], b['splits'])
        groups = [{i['id'] for i in items} for items in a['splits'].values()]
        self.assertEqual([8, 1, 1], [len(g) for g in groups])
        self.assertEqual(set(range(10)), set.union(*groups))
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        for label in self.output.glob('*/labels/*.txt'):
            self.assertEqual('' if label.stem == '0' else '0 0.20000000 0.50000000 0.20000000 0.50000000\n', label.read_text())
        self.assertIn('foil: \\"metal\\"', (self.output / 'data.yaml').read_text())

    def test_geometric_clipping(self):
        self.assertEqual((.1, .15, .2, .3), yolo_bbox([-10, -10, 30, 40], 100, 100))
        for box in ([110, 0, 10, 10], [0, 0, -1, 2], [0, 0, float('nan'), 1], [1, 2, 3]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                yolo_bbox(box, 100, 100)

    def test_missing_image_fails_before_writing(self):
        (self.images / '2.jpg').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            self.convert()
        self.assertFalse(self.output.exists())

    def test_duplicate_content_rejected(self):
        (self.images / '1.jpg').write_bytes((self.images / '0.jpg').read_bytes())
        with self.assertRaisesRegex(ValueError, 'Duplicate image'):
            self.convert()
        self.assertFalse(self.output.exists())

    def test_existing_output_never_overwritten(self):
        self.convert()
        before = (self.output / 'manifest.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            self.convert()
        self.assertEqual(before, (self.output / 'manifest.json').read_bytes())

    def test_bad_category_and_duplicate_image_id_rejected(self):
        self.data['annotations'][0]['category_id'] = 999
        with self.assertRaisesRegex(ValueError, 'unknown'):
            self.convert()
        self.data['annotations'][0]['category_id'] = 7
        self.data['images'][0]['id'] = 1
        with self.assertRaisesRegex(ValueError, 'unique IDs'):
            self.convert()

    def test_path_escape_and_symlink_rejected(self):
        for name in ('../annotations.json', '/etc/passwd', 'C:\\data\\image.jpg', 'a/../../file.jpg'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                image_path(self.images, name)
        target = self.root / 'outside.jpg'
        target.write_bytes(b'synthetic')
        try:
            (self.images / 'link.jpg').symlink_to(target)
        except OSError:
            self.skipTest('OS does not permit symlink creation')
        with self.assertRaises(ValueError):
            image_path(self.images, 'link.jpg')

    def test_crowd_and_unlabelled_images_keep_empty_labels(self):
        self.data['annotations'][0]['iscrowd'] = 1
        result = self.convert()
        self.assertEqual(1, result['crowd_annotations_skipped'])
        self.assertEqual('', next(self.output.glob('*/labels/1.txt')).read_text())

    def test_prepared_dataset_validator_catches_malformed_labels(self):
        from src.utils import check_dataset
        self.convert()
        counts = check_dataset(self.output)
        self.assertEqual(10, sum(split['images'] for split in counts.values()))
        label = next(self.output.glob('train/labels/*.txt'))
        label.write_text('0 0.1 0.5 0.9 1\n')
        with self.assertRaisesRegex(ValueError, 'outside image'):
            check_dataset(self.output)
