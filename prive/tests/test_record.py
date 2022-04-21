import unittest
from prive.datasets import TabularRecord
from prive.datasets import TabularDataset

class TestTabularRecord(unittest.TestCase):
    def setUp(self):
        self.dataset = TabularDataset.read('tests/data/test_texas')
        self.row = self.dataset.get_records([0])
        self.record = TabularRecord(self.row)

    def test_something(self):
        self.assertEqual(self.record.data.shape[0], 1)  # add assertion here


if __name__ == '__main__':
    unittest.main()
