import networkx as nx
import os
import numpy as np
import pandas as pd
import pickle

from .dataset import Dataset, download_url, extract_zip
from .network_description import TUDatasetDescription
from .utils import index_split


def _process(name, filepath):
    """
    Parse files into a TUDataset
    """
    print('Loading TU graph dataset: ' + str(name))
    graph = nx.Graph()

    # add edges
    data_adj = np.loadtxt(
        os.path.join(filepath, '{}_A.txt'.format(name)), delimiter=',').astype(int)
    data_tuple = list(map(tuple, data_adj))
    graph.add_edges_from(data_tuple)

    # add edge labels
    f = os.path.join(filepath, '{}_edge_labels.txt'.format(name))
    if os.path.exists(f):
        data_edge_labels = np.loadtxt(f, delimiter=',').astype(int).tolist()
        nx.set_edge_attributes(graph, dict(zip(data_tuple, data_edge_labels)), "label")

    # add edge attributes
    f = os.path.join(filepath, '{}_edge_attributes.txt'.format(name))
    if os.path.exists(f):
        data_edge_attributes = np.loadtxt(f, delimiter=',').tolist()
        nx.set_edge_attributes(graph, dict(zip(data_tuple, data_edge_attributes)), "attribute")

    # print(nx.get_edge_attributes(graph, "label"))
    # print(nx.get_edge_attributes(graph, "attribute"))
    # print(graph)

    # add nodes, their labels and attributes
    data_node_label = np.loadtxt(
        os.path.join(filepath, '{}_node_labels.txt'.format(name)),
        delimiter=',').astype(int).tolist()

    f = os.path.join(filepath, '{}_node_attributes.txt'.format(name))
    has_node_attr = False
    if os.path.exists(f):
        data_node_attribute = np.loadtxt(f, delimiter=',').tolist()
        has_node_attr = True;

    for i in range(len(data_node_label)):
        graph.add_node(i + 1, label=data_node_label[i],
                       attribute=data_node_attribute[i] if has_node_attr else None)

    # print(nx.get_node_attributes(graph, "label"))
    # print(nx.get_node_attributes(graph, "attribute"))

    # remove isolated nodes and self-loop edges
    # print('Removing isolated nodes and self-loop edges' + str(name))
    # graph.remove_nodes_from(list(nx.isolates(graph)))
    # graph.remove_edges_from(nx.selfloop_edges(graph))

    data_graph_indicator = np.loadtxt(
        os.path.join(filepath, '{}_graph_indicator.txt'.format(name)),
        delimiter=',').astype(int)

    data_graph_label = np.loadtxt(
        os.path.join(filepath, '{}_graph_labels.txt'.format(name)),
        delimiter=',').astype(int)

    has_graph_attr = False
    f = os.path.join(filepath, '{}_graph_attributes.txt'.format(name))
    if os.path.exists(f):
        data_graph_attribute = np.loadtxt(f, delimiter=',')
        has_graph_attr = True

    # split into sub-graphs using graph indicator
    graphs = []
    for idx in range(data_graph_indicator.max()):
        node_idx = np.where(data_graph_indicator == (idx + 1))
        node_list = [x + 1 for x in node_idx[0]]

        sub_graph = graph.subgraph(node_list)
        sub_graph.graph['label'] = data_graph_label[idx]
        sub_graph.graph['attribute'] = data_graph_attribute[idx] if has_graph_attr else None
        graphs.append(sub_graph)

    # pandas.Series of networkx objects
    graphs = pd.Series(graphs)
    description = TUDatasetDescription(label=name)

    return TUDataset(graphs, description)


class TUDataset(Dataset):
    """
    Class to represent TU network data as a Dataset. Internally, the data
    is stored as Pandas Series of networkx objects.

    """

    _url = r'https://www.chrsmrrs.com/graphkerneldatasets/'

    def __init__(self, data, description):
        """
        Parameters
        ----------
        data: pandas.Series
        description: tapas.datasets.network_description.TUDatasetDescription
        """
        self.data = data

        assert isinstance(description, TUDatasetDescription), 'description needs to be of class DataDescription'
        self.description = description

    @classmethod
    def read(cls, name, root):
        """
        Process TU files

        Parameters
        ----------
        name: str
            The name of the dataset.
        root: str
            root directory where the dataset is saved

        Returns
        -------
        TUDataset
            A TUDataset.

        """
        return _process(name, root)

    @classmethod
    def download_and_read(cls, name, root=None):

        """
        Download and process TU files

        Parameters
        ----------
        name: str
            The name of the dataset.
        root: str
            root directory where the dataset to be saved

        Returns
        -------
        TUDataset
            A TUDataset.

        """
        if root is None: root = "./"
        filepath = download_url(f'{cls._url}/{name}.zip', root)
        extract_zip(filepath, root)

        filepath = os.path.join(root, name)
        return _process(name, filepath)

    def write(self, output_path):
        with open(output_path, "wb") as f:
            pickle.dump(self.data, f)

    def read_from_string(self, data, schema):
        pass

    def write_to_string(self):
        pass

    def sample(self, n_samples=1, frac=None, random_state=None):
        """
        Sample a set of records from a TUDataset object.

        Parameters
        ----------
        n_samples : int
            Number of records to sample. If frac is not None, this parameter is ignored.

        frac : float
            Fraction of records to sample.

        Returns
        -------
        TUDataset
            A TUDataset object with a sample of the records of the original object.

        """
        if frac:
            n_samples = int(frac * len(self))

        return TUDataset(
            data=self.data.sample(n_samples, random_state=random_state),
            description=self.description
        )

    def get_records(self, record_ids):
        """
        Get a record from the TUDataset object

        Parameters
        ----------
        record_ids : list[int]
            List of indexes of records to retrieve.

        Returns
        -------
        TabularDataset
            A TUDataset object with the record(s).

        """
        if len(record_ids) == 1:
            return TURecord(
                self.data.iloc[record_ids], self.description, record_ids[0]
            )
        return TUDataset(self.data.iloc[record_ids], self.description)

    def drop_records(self, record_ids=[], n=1, in_place=False):
        """
        Drop records from the TUDataset object, if record_ids is empty it will drop a random record.

        Parameters
        ----------
        record_ids : list[int]
            List of indexes of records to drop.
        n : int
            Number of random records to drop if record_ids is empty.
        in_place : bool
            Bool indicating whether or not to change the dataset in-place or return
            a copy. If True, the dataset is changed in-place. The default is False.

        Returns
        -------
        TUDataset or None
            A new TUDataset object without the record(s) or None if in_place=True.

        """
        if len(record_ids) == 0:
            record_ids = np.random.choice(
                self.data.index, size=n, replace=False).tolist()
        else:
            record_ids = [self.data.index[i] for i in record_ids]

        new_data = self.data.drop(record_ids)

        if in_place:
            self.data = new_data
            return TUDataset(self.data, self.description)

        return TUDataset(new_data, self.description)

    def add_records(self, records, in_place=False):
        """
        Add record(s) to dataset and return modified dataset.

        Parameters
        ----------
        records : TUDataset
            A TUDataset object with the record(s) to add.
        in_place : bool
            Bool indicating whether or not to change the dataset in-place or return
            a copy. If True, the dataset is changed in-place. The default is False.

        Returns
        -------
        TUDataset or None
            A new TUDataset object with the record(s) or None if inplace=True.

        """

        if in_place:
            assert (
                self.description == records.description
            ), "Both datasets must have the same data description"

            self.data = pd.concat([self.data, records.data])
            return

        # if not in_place this does the same as the __add__
        return self.__add__(records)

    def replace(self, records_in, records_out=[], in_place=False):
        """
        Replace a record with another one in the dataset, if records_out is empty it will remove a random record.

        Parameters
        ----------
        records_in : TUDataset
            A TUDataset object with the record(s) to add.
        records_out : list(int)
            List of indexes of records to drop.
        in_place : bool
            Bool indicating whether or not to change the dataset in-place or return
            a copy. If True, the dataset is changed in-place. The default is False.

        Returns
        -------
        TUDataset or None
            A modified TUDataset object with the replaced record(s) or None if in_place=True..

        """
        if len(records_out) > 0:
            assert len(records_out) == len(
                records_in
            ), f"Number of records out must equal number of records in, got {len(records_out)}, {len(records_in)}"

        if in_place:
            self.drop_records(records_out, n=len(records_in), in_place=in_place)
            self.add_records(records_in, in_place=in_place)
            return

        # pass n as a back-up in case records_out=[]
        reduced_dataset = self.drop_records(records_out, n=len(records_in))

        return reduced_dataset.add_records(records_in)

    def create_subsets(self, n, sample_size, drop_records=None):
        """
        Create a number n of subsets of this dataset of size sample_size without
        replacement. If needed, the records can be dropped from this dataset.

        Parameters
        ----------
        n : int
            Number of datasets to create.
        sample_size : int
            Size of the subset datasets to be created.
        drop_records: bool
            Whether to remove the records sampled from this dataset (in place).

        Returns
        -------
        list(TabularDataset)
            A lists containing subsets of the data with and without the target record(s).

        """
        assert sample_size <= len(
            self
        ), f"Cannot create subsets larger than original dataset, sample_size max: {len(self)} got {sample_size}"

        # Create splits.
        splits = index_split(self.data.shape[0], sample_size, n)

        # Returns a list of TabularDataset subsampled from this dataset.
        subsamples = [self.get_records(train_index) for train_index in splits]

        # If required, remove the indices from the dataset.
        if drop_records:
            for train_index in splits:
                self.drop_records(train_index, in_place=True)

        return subsamples

    def empty(self):
        """
        Create an empty TUDataset with the same description as the current one.
        Short-hand for TUDataset.get_records([]).

        Returns
        -------
        TUDataset
            Empty tudataset.

        """
        return self.get_records([])

    def copy(self):
        """
        Create a TabularDataset that is a deep copy of this one. In particular,
        the underlying data is copied and can thus be modified freely.

        Returns
        -------
        TabularDataset
            A copy of this TabularDataset.

        """
        return TUDataset(self.data.copy(), self.description)

    def view(self, columns = None, exclude_columns = None):
        pass

    def __add__(self, other):
        assert (
            self.description == other.description
        ), "Both datasets must have the same data description"

        return TUDataset(pd.concat([self.data, other.data]), self.description)

    def __iter__(self):
        """
        Returns an iterator over records in this dataset,

        Returns
        -------
        iterator
            An iterator object that iterates over individual records, as TURecords.

        """
        # iterrows() returns tuples (index, record), and map applies a 1-argument
        # function to the iterable it is given, hence why we have idx_and_rec
        # instead of the cleaner (idx, rec).

        convert_record = lambda idx_and_rec: TURecord.from_dataset(
            TUDataset(
                data=pd.Series([idx_and_rec[1]]),
                description=self.description
            )
        )
        return map(convert_record, self.data.iteritems())

    def __len__(self):
        """
        Returns the number of records in this dataset.

        Returns
        -------
        integer
            length: number of records in this dataset.

        """
        return self.data.shape[0]

    def __contains__(self, item):
        """
        Determines the truth value of `item in self`. The only items considered
        to be in a TUDataset are the rows, treated as 1-row TUDataset.

        Parameters
        ----------
        item : Object
            Object to check membership of.

        Returns
        -------
        bool
            Whether or not item is considered to be contained in self.

        """
        if not isinstance(item, TUDataset):
            raise ValueError(
                f"Only TabularDatasets can be checked for containment, not {type(item)}"
            )
        if len(item) != 1:
            raise ValueError(
                f"Only length-1 TUDataset can be checked for containment, got length {len(item)})"
            )

        return (self.data == item.data.iloc[0]).all(axis=1).any()

    @property
    def label(self):
        return self.description.label


class TURecord(TUDataset):
    """
    Class for TU record object. The TU data is a 1D array

    """

    def __init__(self, data, description, identifier):
        super().__init__(data, description)
        # id of the object based on their index on the original dataset
        self.id = identifier

    @classmethod
    def from_dataset(cls, tu_dataset):
        """
        Create a TUDataset object from a TUDataset object containing 1 record.

        Parameters
        ----------
        tu_dataset: TUDataset
            A TUDataset object containing one record.

        Returns
        -------
        TUDataset
            A TUDataset object

        """
        if tu_dataset.data.shape[0] != 1:
            raise AssertionError(
                f"Parent TUDataset object must contain only 1 record, not {tu_dataset.data.shape[0]}"
            )

        return cls(
            tu_dataset.data, tu_dataset.description, tu_dataset.data.index.values[0]
        )

    def get_id(self, tu_dataset):
        """

        Check if the record is found on a given TabularDataset and return the object id (index) on that
        dataset.

        Parameters
        ----------
        tu_dataset: TabularDataset
            A TabularDataset object.

        Returns
        -------
        int
            The id of the object based on the index in the original dataset.

        """

        merged = pd.merge(tu_dataset.data, self.data, how="outer", indicator=True)

        if merged[merged["_merge"] == "both"].shape[0] != 1:
            raise AssertionError(
                "Error, more than one copy of this record is present on the dataset"
            )

        return merged[merged["_merge"] == "both"].index.values[0]

    def set_id(self, identifier):
        """
        Overwrite the id attribute on the TabularRecord object.

        Parameters
        ----------
        identifier: int or str
            An id value to be assigned to the TabularRecord id attribute

        Returns
        -------
        None

        """
        self.id = identifier
        self.data.index = pd.Index([identifier])

        return

    def set_value(self, column, value):
        """
        Overwrite the value of attribute `column` of the TabularRecord object.

        Parameters
        ----------
        column: str
            The identifier of the attribute to be replaced.
        value: (value set of column)
            The value to set the `column` of the record.

        Returns
        -------
        None

        """
        self.data[column] = value

    def copy(self):
        """
        Create a TabularRecord that is a deep copy of this one. In particular,
        the underlying data is copied and can thus be modified freely.

        Returns
        -------
        TabularRecord
            A copy of this TabularRecord.

        """
        return TURecord(self.data.copy(), self.description, self.id)

    @property
    def label(self):
        """
        The label for records is their identifier. We assume here that the label
        of the rest of the dataset is obvious from context. If not, it can be
        retrived as self.description.label.

        """
        return str(self.id)
