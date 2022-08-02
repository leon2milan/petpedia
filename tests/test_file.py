import unittest
from pathlib import Path


class FileTest(unittest.TestCase):
    def setUp(self):
        self.files = [
            'data/dictionary/segmentation/custom.txt',
            'models/intent/detector.txt'
        ]

    def test_predict(self):
        expect_output = [152357, 175212]
        output = []
        for x in self.files:
            output.append(Path(x).stat().st_size)
        self.assertEqual(output, expect_output)
