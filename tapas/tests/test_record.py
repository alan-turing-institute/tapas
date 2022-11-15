import unittest

from tapas.datasets import TabularRecord
from tapas.datasets import TabularDataset


class TestTabularRecord(unittest.TestCase):
    def setUp(self):
        self.id = 10
        self.tabulardataset = TabularDataset.read('tests/data/test_texas')
        self.row = self.tabulardataset.get_records([self.id])
        self.record = TabularRecord.from_dataset(self.row)

    def test_init(self):
        self.assertEqual(self.record.data.shape[0], 1)
        # check if it conserves the original index value
        self.assertEqual(self.record.id, self.id)

    def test_get_id(self):
        value = self.record.get_id(self.tabulardataset)

        self.assertEqual(value, 10)

    def test_set_id(self):
        new_id = '100'
        self.record.set_id(new_id)

        self.assertEqual(self.record.id, new_id)
        self.assertEqual(self.record.data.index.values, new_id)


if __name__ == '__main__':
    unittest.main()
