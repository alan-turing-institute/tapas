"""A  simple test for the dataset class"""
from unittest import TestCase

from warnings import filterwarnings

filterwarnings('ignore')

from privE.datasets.dataset import TabularDataset


class TestTabularDataset(TestCase):
    def test(self):
        data = TabularDataset('tests/data/texas')
        self.assertEqual(data.dataset.shape[0], 999)
