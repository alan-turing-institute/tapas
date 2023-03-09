"""Loads and represents graph datasets from the TU repository.

A TUDataset is a collection of graphs such that each graph is the "record" of
a natural person. Such a graph could be, e.g., the k-IIG of a person in a
social network, or internal cash flows of a company.

"""

import networkx as nx
import numpy as np
import pandas as pd
import pickle
import os
import requests
import zipfile

from .dataset import RecordSetDataset, Record, DataDescription
from .utils import index_split


## Helper functions for TU Datasets.

def _download_url(url, fp):
    """
    Download the content (a TU dataset) from a given url.

    """
    if os.path.isdir(fp):
        path = os.path.join(fp, url.split("/")[-1])
    else:
        path = fp

    print("Downloading %s from %s..." % (path, url))

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError("Failed downloading url %s" % url)
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

    return path


def _process(name, filepath):
    """
    Parse files in TU format into a TUDataset object.

    """
    print("Loading TU graph dataset: " + str(name))
    graph = nx.Graph()

    # add edges
    data_adj = np.loadtxt(
        os.path.join(filepath, "{}_A.txt".format(name)), delimiter=","
    ).astype(int)
    data_tuple = list(map(tuple, data_adj))
    graph.add_edges_from(data_tuple)

    # add edge labels
    f = os.path.join(filepath, "{}_edge_labels.txt".format(name))
    if os.path.exists(f):
        data_edge_labels = np.loadtxt(f, delimiter=",").astype(int).tolist()
        nx.set_edge_attributes(graph, dict(zip(data_tuple, data_edge_labels)), "label")

    # add edge attributes
    f = os.path.join(filepath, "{}_edge_attributes.txt".format(name))
    if os.path.exists(f):
        data_edge_attributes = np.loadtxt(f, delimiter=",").tolist()
        nx.set_edge_attributes(
            graph, dict(zip(data_tuple, data_edge_attributes)), "attribute"
        )

    # add nodes, their labels and attributes
    data_node_label = (
        np.loadtxt(
            os.path.join(filepath, "{}_node_labels.txt".format(name)), delimiter=","
        )
        .astype(int)
        .tolist()
    )

    f = os.path.join(filepath, "{}_node_attributes.txt".format(name))
    has_node_attr = False
    if os.path.exists(f):
        data_node_attribute = np.loadtxt(f, delimiter=",").tolist()
        has_node_attr = True

    for i in range(len(data_node_label)):
        graph.add_node(
            i + 1,
            label=data_node_label[i],
            attribute=data_node_attribute[i] if has_node_attr else None,
        )

    data_graph_indicator = np.loadtxt(
        os.path.join(filepath, "{}_graph_indicator.txt".format(name)), delimiter=","
    ).astype(int)

    data_graph_label = np.loadtxt(
        os.path.join(filepath, "{}_graph_labels.txt".format(name)), delimiter=","
    ).astype(int)

    has_graph_attr = False
    f = os.path.join(filepath, "{}_graph_attributes.txt".format(name))
    if os.path.exists(f):
        data_graph_attribute = np.loadtxt(f, delimiter=",")
        has_graph_attr = True

    # split into sub-graphs using graph indicator
    graphs = []
    for idx in range(data_graph_indicator.max()):
        node_idx = np.where(data_graph_indicator == (idx + 1))
        node_list = [x + 1 for x in node_idx[0]]

        sub_graph = graph.subgraph(node_list)
        sub_graph.graph["label"] = data_graph_label[idx]
        sub_graph.graph["attribute"] = (
            data_graph_attribute[idx] if has_graph_attr else None
        )
        graphs.append(sub_graph)

    # pandas.Series of networkx objects
    graphs = pd.Series(graphs)
    description = TUDatasetDescription(label=name)

    return TUDataset(graphs, description)


## Instances of the different classes for TU datasets.


# The description is identical.
class TUDatasetDescription(DataDescription):
    """
    TU Datasets all have the same format. This description class thus only
    contains a label. This is required for compatibility with other classes,
    i.e. threat models and some of the functionalities of TabularDatasets.

    """


class TUDataset(RecordSetDataset):
    """
    Class to represent TU network data as a Dataset. Internally, the data
    is stored as Pandas Series of networkx objects.

    """

    _url = r"https://www.chrsmrrs.com/graphkerneldatasets/"

    def __init__(self, data, description):
        """
        Parameters
        ----------
        data: pandas.Series
        description: tapas.datasets.TUDatasetDescription

        """
        assert isinstance(
            description, TUDatasetDescription
        ), "description needs to be of class DataDescription"
        # Ininitalise the parent class with the correct record class.
        RecordSetDataset.__init__(self, data, description, TURecord)

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
        if root is None:
            root = "./"
        filepath = _download_url(f"{cls._url}/{name}.zip", root)
        with zipfile.ZipFile(filepath, "r") as f:
            f.extractall(root)

        filepath = os.path.join(root, name)
        return _process(name, filepath)

    def write(self, output_path):
        """
        Write dataset as object to file.

        """
        with open(output_path, "wb") as f:
            pickle.dump(self.data, f)

    def read_from_string(self, data, schema):
        raise NotimplementedError()

    def write_to_string(self):
        raise NotimplementedError()

    # def sample(self, n_samples=1, frac=None, random_state=None):
    #     """
    #     Sample a set of records from a TUDataset object.
    #     Note that a record in TUDataset means a graph.

    #     Parameters
    #     ----------
    #     n_samples : int
    #         Number of records to sample. If frac is not None, this parameter is ignored.

    #     frac : float
    #         Fraction of records to sample.

    #     random_state : optional
    #         Passed to `pandas.DataFrame.sample()`

    #     Returns
    #     -------
    #     TUDataset
    #         A TUDataset object with a sample of the records of the original object.

    #     """
    #     if frac:
    #         n_samples = int(frac * len(self))

    #     return TUDataset(
    #         data=self.data.sample(n_samples, random_state=random_state),
    #         description=self.description,
    #     )

    # def get_records(self, record_ids):
    #     """
    #     Get a record from the TUDataset object

    #     Parameters
    #     ----------
    #     record_ids : list[int]
    #         List of indexes of records to retrieve.

    #     Returns
    #     -------
    #     TUDataset
    #         A TUDataset object with the record(s).

    #     """
    #     if len(record_ids) == 1:
    #         return TURecord(self.data.iloc[record_ids], self.description, record_ids[0])
    #     return TUDataset(self.data.iloc[record_ids], self.description)

    # def drop_records(self, record_ids=[], n=1, in_place=False):
    #     """
    #     Drop records from the TUDataset object, if record_ids is empty it will drop a random record.

    #     Parameters
    #     ----------
    #     record_ids : list[int]
    #         List of indexes of records to drop.
    #     n : int
    #         Number of random records to drop if record_ids is empty.
    #     in_place : bool
    #         Bool indicating whether or not to change the dataset in-place or return
    #         a copy. If True, the dataset is changed in-place. The default is False.

    #     Returns
    #     -------
    #     TUDataset or None
    #         A new TUDataset object without the record(s) or None if in_place=True.

    #     """
    #     if len(record_ids) == 0:
    #         record_ids = np.random.choice(
    #             self.data.index, size=n, replace=False
    #         ).tolist()
    #     else:
    #         record_ids = [self.data.index[i] for i in record_ids]

    #     new_data = self.data.drop(record_ids)

    #     if in_place:
    #         self.data = new_data
    #         return TUDataset(self.data, self.description)

    #     return TUDataset(new_data, self.description)

    # def add_records(self, records, in_place=False):
    #     """
    #     Add record(s) to dataset and return modified dataset.

    #     Parameters
    #     ----------
    #     records : TUDataset
    #         A TUDataset object with the record(s) to add.
    #     in_place : bool
    #         Bool indicating whether or not to change the dataset in-place or return
    #         a copy. If True, the dataset is changed in-place. The default is False.

    #     Returns
    #     -------
    #     TUDataset or None
    #         A new TUDataset object with the record(s) or None if inplace=True.

    #     """

    #     if in_place:
    #         assert (
    #             self.description == records.description
    #         ), "Both datasets must have the same data description"

    #         self.data = pd.concat([self.data, records.data])
    #         return

    #     # if not in_place this does the same as the __add__
    #     return self.__add__(records)

    # def replace(self, records_in, records_out=[], in_place=False):
    #     """
    #     Replace a record with another one in the dataset, if records_out is empty it will remove a random record.

    #     Parameters
    #     ----------
    #     records_in : TUDataset
    #         A TUDataset object with the record(s) to add.
    #     records_out : list(int)
    #         List of indexes of records to drop.
    #     in_place : bool
    #         Bool indicating whether or not to change the dataset in-place or return
    #         a copy. If True, the dataset is changed in-place. The default is False.

    #     Returns
    #     -------
    #     TUDataset or None
    #         A modified TUDataset object with the replaced record(s) or None if in_place=True..

    #     """
    #     if len(records_out) > 0:
    #         assert len(records_out) == len(
    #             records_in
    #         ), f"Number of records out must equal number of records in, got {len(records_out)}, {len(records_in)}"

    #     if in_place:
    #         self.drop_records(records_out, n=len(records_in), in_place=in_place)
    #         self.add_records(records_in, in_place=in_place)
    #         return

    #     # pass n as a back-up in case records_out=[]
    #     reduced_dataset = self.drop_records(records_out, n=len(records_in))

    #     return reduced_dataset.add_records(records_in)

    # def create_subsets(self, n, sample_size, drop_records=None):
    #     """
    #     Create a number n of subsets of this dataset of size sample_size without
    #     replacement. If needed, the records can be dropped from this dataset.

    #     Parameters
    #     ----------
    #     n : int
    #         Number of datasets to create.
    #     sample_size : int
    #         Size of the subset datasets to be created.
    #     drop_records: bool
    #         Whether to remove the records sampled from this dataset (in place).

    #     Returns
    #     -------
    #     list(TUDataset)
    #         A lists containing subsets of the data with and without the target record(s).

    #     """
    #     assert sample_size <= len(
    #         self
    #     ), f"Cannot create subsets larger than original dataset, sample_size max: {len(self)} got {sample_size}"

    #     # Create splits.
    #     splits = index_split(self.data.shape[0], sample_size, n)

    #     # Returns a list of TUDataset subsampled from this dataset.
    #     subsamples = [self.get_records(train_index) for train_index in splits]

    #     # If required, remove the indices from the dataset.
    #     if drop_records:
    #         for train_index in splits:
    #             self.drop_records(train_index, in_place=True)

    #     return subsamples

    # def empty(self):
    #     """
    #     Create an empty TUDataset with the same description as the current one.
    #     Short-hand for TUDataset.get_records([]).

    #     Returns
    #     -------
    #     TUDataset
    #         Empty tudataset.

    #     """
    #     return self.get_records([])

    # def copy(self):
    #     """
    #     Create a TUDataset that is a deep copy of this one. In particular,
    #     the underlying data is copied and can thus be modified freely.

    #     Returns
    #     -------
    #     TUDataset
    #         A copy of this TUDataset.

    #     """
    #     return TUDataset(self.data.copy(), self.description)


    # def __add__(self, other):
    #     assert (
    #         self.description == other.description
    #     ), "Both datasets must have the same data description"

    #     return TUDataset(pd.concat([self.data, other.data]), self.description)

    # def __iter__(self):
    #     """
    #     Returns an iterator over records in this dataset,

    #     Returns
    #     -------
    #     iterator
    #         An iterator object that iterates over individual records, as TURecords.

    #     """
    #     # iterrows() returns tuples (index, record), and map applies a 1-argument
    #     # function to the iterable it is given, hence why we have idx_and_rec
    #     # instead of the cleaner (idx, rec).

    #     convert_record = lambda idx_and_rec: TURecord.from_dataset(
    #         TUDataset(data=pd.Series([idx_and_rec[1]]), description=self.description)
    #     )
    #     return map(convert_record, self.data.items())

    # def __len__(self):
    #     """
    #     Returns the number of records in this dataset.

    #     Returns
    #     -------
    #     integer
    #         length: number of records in this dataset.

    #     """
    #     return self.data.shape[0]

    # def __contains__(self, item):
    #     """
    #     Determines the truth value of `item in self`. The only items considered
    #     to be in a TUDataset are the rows, treated as 1-row TUDataset.

    #     Parameters
    #     ----------
    #     item : Object
    #         Object to check membership of.

    #     Returns
    #     -------
    #     bool
    #         Whether or not item is considered to be contained in self.

    #     """
    #     if not isinstance(item, TUDataset):
    #         raise ValueError(
    #             f"Only TUDatasets can be checked for containment, not {type(item)}"
    #         )
    #     if len(item) != 1:
    #         raise ValueError(
    #             f"Only length-1 TUDataset can be checked for containment, got length {len(item)})"
    #         )

    #     return (self.data == item.data.iloc[0]).any()

    # @property
    # def label(self):
    #     return self.description.label


class TURecord(Record, TUDataset):
    """
    Class for TU record object. The TU data is a 1D array

    """

    def __init__(self, data, description, identifier):
        Record.__init__(self, data, description, identifier)
        TUDataset.__init__(self, data, description)

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

