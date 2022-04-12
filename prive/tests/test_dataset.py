"""A  simple test for the dataset class"""
from unittest import TestCase

from warnings import filterwarnings

filterwarnings('ignore')

from prive.datasets import TabularDataset


class TestTabularDataset(TestCase):
    def test_read(self):
        data = TabularDataset.read('tests/data/texas')
        self.assertEqual(data.dataset.shape[0], 999)

    def test_sample(self):
        data = TabularDataset.read('tests/data/texas')

        # returns a subset of the samples
        data_sample = data.sample(500)

        self.assertEqual(data_sample.description, data.description)
        self.assertEqual(data_sample.dataset.shape[0], 500)

    def test_add(self):
        data = TabularDataset.read('tests/data/texas')

        # returns a subset of the samples
        data_sample1 = data.sample(500)
        data_sample2 = data.sample(500)

        data_1000 = data_sample1 + data_sample2

        self.assertEqual(data_1000.description, data_sample1.description)
        self.assertEqual(data_1000.dataset.shape[0], 1000)

if __name__ == '__main__':
    unittest.main()
