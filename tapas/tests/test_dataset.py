"""A  simple test for the dataset class"""
import copy
from random import randint
import json
from unittest import TestCase
from warnings import filterwarnings
import networkx as nx

filterwarnings("ignore")

from tapas.datasets import TabularDataset, TabularRecord, TUDataset, TURecord
from tapas.datasets.data_description import DataDescription
from tapas.datasets.canary import create_canary


class TestDescription(TestCase):
    def test_description_equal(self):
        dummy_descr = [
            {"name": "a", "type": "finite", "representation": ["A", "B", "C"]},
            {"name": "B", "type": "countable", "representation": "integer"},
            {"name": "   ", "type": "finite", "representation": 10},
        ]
        description_1 = DataDescription(dummy_descr)
        description_2 = DataDescription(copy.deepcopy(dummy_descr))
        self.assertEqual(description_1, description_2)


class TestTUDataset(TestCase):
    def setUp(self):
        self.tu_dataset = TUDataset.download_and_read("Cuneiform", "data/network")
        self.tu_dataset_in = TUDataset.read("in", "data/network")

    def test_read(self):
        self.assertEqual(len(self.tu_dataset_in.data), 5)
        self.assertEqual(len(self.tu_dataset.data), 267)

        # check subgraph node number (graph indicator)
        graph_0 = self.tu_dataset.data[0]
        graph_5 = self.tu_dataset.data[4]
        self.assertEqual(len(graph_0), 36)
        self.assertEqual(len(graph_5), 16)

        # check edge label and attribute
        label = nx.get_edge_attributes(graph_5, "label")
        self.assertEqual(label[97, 89], 1)
        attr = nx.get_edge_attributes(graph_5, "attribute")
        self.assertEqual(attr[97, 89], [7.231583897539494, 2.5010477503933224])

        # check node label and attribute
        label = nx.get_node_attributes(graph_0, "label")
        self.assertEqual(label[2], [1, 0])
        attr = nx.get_node_attributes(graph_0, "attribute")
        self.assertEqual(attr[1], [3.6595633181952874, 2.6287972093083667, -13.3789])

    def test_sample(self):
        # returns a subset of the samples
        data_sample = self.tu_dataset.sample(20)

        self.assertEqual(data_sample.description, self.tu_dataset.description)
        self.assertEqual(len(data_sample), 20)

    def test_add(self):
        # returns a subset of the samples
        data_sample1 = self.tu_dataset.sample(50)
        data_sample2 = self.tu_dataset.sample(50)

        data_100 = data_sample1 + data_sample2

        self.assertEqual(data_100.description, data_sample1.description)
        self.assertEqual(data_100.data.shape[0], 100)

        # Sample a random index and check that the row in data_100 matches that in data_sample2
        idx = randint(0, 49)
        self.assertTrue(
            data_100.data.iloc[50 + idx] == data_sample2.data.iloc[idx]
        )

    def test_get_records(self):
        # returns a subset of the records
        index = [10, 20]
        records = self.tu_dataset.get_records(index)

        self.assertEqual(records.data.iloc[0], self.tu_dataset.data.iloc[index[0]])

        self.assertEqual(len(records), len(index))

    def test_drop_records(self):
        # returns a subset of the records
        index = [10, 20, 50, 100]
        new_dataset = self.tu_dataset.drop_records(index)

        self.assertEqual(len(new_dataset), len(self.tu_dataset) - len(index))

        # check that the records have been removed
        for idx in index:
            self.assertNotIn(self.tu_dataset.get_records([idx]), new_dataset)

        # check that the index has been properly maintained
        self.assertEqual(new_dataset.data.index[index[0]], index[0] + 1)

        # drop random record
        new_dataset = self.tu_dataset.drop_records()
        self.assertEqual(len(new_dataset), len(self.tu_dataset) - 1)

        new_dataset = self.tu_dataset.drop_records(n=4)
        self.assertEqual(len(new_dataset), len(self.tu_dataset) - 4)

        # test in-place flag
        new_dataset = copy.copy(self.tu_dataset)  # Don't want to modify self.dataset
        new_dataset.drop_records(index, in_place=True)
        # check length
        self.assertEqual(len(new_dataset), len(self.tu_dataset) - len(index))
        # check records are gone
        for idx in index:
            self.assertNotIn(idx, new_dataset.data.index)

    def test_create_subsets(self):
        # returns a subset of the records
        index = [100]

        rI = self.tu_dataset.create_subsets(10, 100)

        self.assertEqual(len(rI), 10)
        self.assertEqual(100, rI[0].data.shape[0])

    def test_replace(self):
        # returns a subset of the records
        index = [100, 200]
        records_in = self.tu_dataset.get_records(index)

        # returns a subset of the samples
        data_sample100 = self.tu_dataset.get_records([i for i in range(100)])

        index_to_drop = [20, 30]

        replaced_sample = data_sample100.replace(records_in, index_to_drop)

        self.assertEqual(len(data_sample100), len(replaced_sample))

        # check if record is in dataset
        self.assertFalse(
            (
                replaced_sample.data
                == self.tu_dataset.get_records(index_to_drop).data.iloc[0]
            )
            .all()
            .all()
        )

        # check removing random record
        replaced_sample = data_sample100.replace(records_in)
        self.assertEqual(len(data_sample100), len(replaced_sample))

        # check in-place flag
        new_dataset = copy.copy(data_sample100)
        new_dataset.replace(records_in, index_to_drop, in_place=True)

        self.assertEqual(len(new_dataset), len(data_sample100))
        # check records were removed
        for idx in index_to_drop:
            self.assertNotIn(data_sample100.get_records([idx]), new_dataset)
        # check records were added
        for idx in index:
            self.assertIn(self.tu_dataset.get_records([idx]), new_dataset)

    def test_empty(self):
        empty_dataset = self.tu_dataset.empty()
        self.assertEqual(len(empty_dataset), 0)
        self.assertEqual(empty_dataset.description, self.tu_dataset.description)

    def test_iter(self):
        record_count = 0
        for record in self.tu_dataset:
            self.assertEqual(type(record), TURecord)
            self.assertEqual(record.description, self.tu_dataset.description)
            self.assertEqual(record.data.shape[0], 1)
            record_count += 1
        self.assertEqual(record_count, len(self.tu_dataset))

    def test_contains(self):
        indices = [15, 30]
        records = self.tu_dataset.get_records(indices)

        for index, graph in enumerate(records):
            self.assertIn(graph, self.tu_dataset)

    # def test_canary(self):
    #     new_dataset, canary = create_canary(self.tu_dataset)
    #     self.assertEqual(new_dataset.description, canary.description)

    def test_add_records(self):
        # returns a subset of the records
        index = [100]
        record = self.tu_dataset.get_records(index)

        new_dataset = self.tu_dataset.add_records(record)

        self.assertEqual(
            new_dataset.data.shape[0], self.tu_dataset.data.shape[0] + len(index)
        )

        # test in-place flag
        new_dataset = copy.copy(self.tu_dataset)
        new_dataset.add_records(record, in_place=True)
        # check length
        self.assertEqual(len(new_dataset), len(self.tu_dataset) + len(index))
        # check records are in
        for idx in index:
            self.assertIn(idx, new_dataset.data.index)


class TestTabularDataset(TestCase):
    def setUp(self):
        self.dataset = TabularDataset.read("data/test_texas")
        self.row_in = TabularDataset.read("data/row_in_texas")
        self.row_out = TabularDataset.read("data/row_out_texas")

    def test_read(self):

        self.assertEqual(len(self.dataset), 998)

        with open("data/test_texas.json") as f:
            description = json.load(f)

        self.assertEqual(self.dataset.description.schema, description)

    def test_sample(self):
        # returns a subset of the samples
        data_sample = self.dataset.sample(500)

        self.assertEqual(data_sample.description, self.dataset.description)
        self.assertEqual(len(data_sample), 500)

    def test_add(self):
        # returns a subset of the samples
        data_sample1 = self.dataset.sample(500)
        data_sample2 = self.dataset.sample(500)

        data_1000 = data_sample1 + data_sample2

        self.assertEqual(data_1000.description, data_sample1.description)
        self.assertEqual(data_1000.data.shape[0], 1000)

        # Sample a random index and check that the row in data_1000 matches that in data_sample2
        idx = randint(0, 499)
        self.assertTrue(
            (data_1000.data.iloc[500 + idx] == data_sample2.data.iloc[idx]).all()
        )

    def test_get_records(self):
        # returns a subset of the records
        index = [10, 20]
        records = self.dataset.get_records(index)

        self.assertEqual(records.data.iloc[0][0], self.dataset.data.iloc[index[0]][0])

        self.assertEqual(len(records), len(index))

    def test_drop_records(self):
        # returns a subset of the records
        index = [10, 20, 50, 100]
        new_dataset = self.dataset.drop_records(index)

        self.assertEqual(len(new_dataset), len(self.dataset) - len(index))

        # check that the records have been removed
        for idx in index:
            self.assertNotIn(self.dataset.get_records([idx]), new_dataset)

        # check that the index has been properly maintained
        self.assertEqual(new_dataset.data.index[index[0]], index[0] + 1)

        # drop random record
        new_dataset = self.dataset.drop_records()
        self.assertEqual(len(new_dataset), len(self.dataset) - 1)

        new_dataset = self.dataset.drop_records(n=4)
        self.assertEqual(len(new_dataset), len(self.dataset) - 4)

        # test in-place flag
        new_dataset = copy.copy(self.dataset)  # Don't want to modify self.dataset
        new_dataset.drop_records(index, in_place=True)
        # check length
        self.assertEqual(len(new_dataset), len(self.dataset) - len(index))
        # check records are gone
        for idx in index:
            self.assertNotIn(idx, new_dataset.data.index)

    def test_add_records(self):
        # returns a subset of the records
        index = [100]
        record = self.dataset.get_records(index)

        new_dataset = self.dataset.add_records(record)

        self.assertEqual(
            new_dataset.data.shape[0], self.dataset.data.shape[0] + len(index)
        )

        # test in-place flag
        new_dataset = copy.copy(self.dataset)  # don't want to modify self.dataset
        new_dataset.add_records(record, in_place=True)
        # check length
        self.assertEqual(len(new_dataset), len(self.dataset) + len(index))
        # check records are in
        for idx in index:
            self.assertIn(idx, new_dataset.data.index)

    def test_create_subsets(self):
        # returns a subset of the records
        index = [100]

        rI = self.dataset.create_subsets(10, 100)

        self.assertEqual(len(rI), 10)
        self.assertEqual(100, rI[0].data.shape[0])

    def test_replace(self):
        # returns a subset of the records
        index = [200, 300]
        records_in = self.dataset.get_records(index)

        # returns a subset of the samples
        data_sample100 = self.dataset.get_records([i for i in range(100)])

        index_to_drop = [20, 30]

        replaced_sample = data_sample100.replace(records_in, index_to_drop)

        self.assertEqual(len(data_sample100), len(replaced_sample))

        # check if record is in dataset
        self.assertFalse(
            (
                replaced_sample.data
                == self.dataset.get_records(index_to_drop).data.iloc[0]
            )
            .all()
            .all()
        )

        # check removing random record
        replaced_sample = data_sample100.replace(records_in)
        self.assertEqual(len(data_sample100), len(replaced_sample))

        # check in-place flag
        new_dataset = copy.copy(data_sample100)  # don't want to modify self.dataset
        new_dataset.replace(records_in, index_to_drop, in_place=True)

        self.assertEqual(len(new_dataset), len(data_sample100))
        # check records were removed
        for idx in index_to_drop:
            self.assertNotIn(data_sample100.get_records([idx]), new_dataset)
        # check records were added
        for idx in index:
            self.assertIn(self.dataset.get_records([idx]), new_dataset)

    def test_empty(self):
        empty_dataset = self.dataset.empty()
        self.assertEqual(len(empty_dataset), 0)
        self.assertEqual(empty_dataset.description, self.dataset.description)

    def test_iter(self):
        record_count = 0
        for record in self.dataset:
            self.assertEqual(type(record), TabularRecord)
            self.assertEqual(record.description, self.dataset.description)
            self.assertEqual(record.data.shape[0], 1)
            record_count += 1
        self.assertEqual(record_count, len(self.dataset))

    def test_contains(self):
        self.assertNotIn(self.row_out, self.dataset)
        self.assertIn(self.row_in, self.dataset)

        indices = [15, 30]
        rows = self.dataset.get_records(indices)
        for row in rows:
            self.assertIn(row, self.dataset)

    def test_canary(self):
        new_dataset, canary = create_canary(self.dataset)
        self.assertEqual(new_dataset.description, canary.description)


if __name__ == "__main__":
    unittest.main()
