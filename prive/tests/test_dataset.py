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

    def test_get_records(self):
        data = TabularDataset.read('tests/data/texas')

        # returns a subset of the records
        index = [10, 20]
        record = data.get_records(index)

        self.assertEqual(record.dataset.iloc[0]['DISCHARGE'], data.dataset.iloc[index[0]]['DISCHARGE'])
        self.assertEqual(record.dataset.shape[0], len(index))

    def test_drop_records(self):
        data = TabularDataset.read('tests/data/texas')

        # returns a subset of the records
        index = [10, 20,50,100]
        new_dataset = data.drop_records(index)

        self.assertEqual(new_dataset.dataset.shape[0], data.dataset.shape[0]-len(index))

        # check is record is in new dataset
        is_record_in_new_dataset = (
                    new_dataset.dataset == data.get_records(index).dataset.iloc[2]).all().all()
        self.assertEqual(is_record_in_new_dataset, False)

        # drop random record
        new_dataset = data.drop_records()
        self.assertEqual(new_dataset.dataset.shape[0], data.dataset.shape[0] - 1)

    def test_add_records(self):
        data = TabularDataset.read('tests/data/texas')

        # returns a subset of the records
        index = [100]
        record = data.get_records(index)

        new_dataset = data.add_records(record)

        self.assertEqual(new_dataset.dataset.shape[0], data.dataset.shape[0] + len(index))



    def test_create_subsets(self):
        data = TabularDataset.read('tests/data/texas')

        # returns a subset of the records
        index = [100]

        rI, rO = data.create_subsets(10, 100, index)

        self.assertEqual(len(rI), len(rO))
        self.assertEqual(rI[0].dataset.shape[0],rO[0].dataset.shape[0]+1)

    def test_replace(self):
        data = TabularDataset.read('tests/data/texas')

        # returns a subset of the records
        index = [200,300]
        records_in = data.get_records(index)

        # returns a subset of the samples
        data_sample100 = data.get_records([i for i in range(100)])

        index_to_drop = [20,30]

        replaced_sample = data_sample100.replace(records_in,index_to_drop)

        self.assertEqual(data_sample100.dataset.shape[0], replaced_sample.dataset.shape[0])

        # check is record is in dataset
        is_record_in_replaced_dataset = (replaced_sample.dataset == data.get_records(index_to_drop).dataset.iloc[0]).all().all()
        self.assertEqual(is_record_in_replaced_dataset,False)

        # check removing random record
        replaced_sample = data_sample100.replace(records_in)
        self.assertEqual(data_sample100.dataset.shape[0]+1, replaced_sample.dataset.shape[0])







if __name__ == '__main__':
    unittest.main()
