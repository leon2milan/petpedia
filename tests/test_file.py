import unittest
from pathlib import Path


class FileTest(unittest.TestCase):
    def setUp(self):
        self.files = ['data/dictionary/segmentation/custom.txt', ]

    def test_predict(self):
        expect_output = [
            102508
        ]
        output = []
        for x in self.files:
            output.append(Path(x).stat().st_size)
        self.assertEquals(output, expect_output)
