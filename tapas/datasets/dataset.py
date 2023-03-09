"""Classes to represent the data object."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .utils import index_split


class Dataset(ABC):
    """
    Base class for the dataset object.

    """

    @abstractmethod
    def read(self, input_path):
        """
        Read dataset and description file.

        """
        pass

    @abstractmethod
    def write(self, output_path):
        """
        Write dataset and description to file.

        """
        pass

    @abstractmethod
    def read_from_string(self, data, schema):
        """
        Read from dataset and description as strings.
        """
        pass

    @abstractmethod
    def write_to_string(self):
        """
        Write dataset to a string.
        """
        pass

    @abstractmethod
    def sample(self, n_samples):
        """
        Sample from dataset a set of records.

        """
        pass

    @abstractmethod
    def get_records(self, record_ids):
        """
        Select and return a record(s).

        """
        pass

    @abstractmethod
    def drop_records(self, record_ids):
        """
        Drop a record(s) and return modified dataset.

        """
        pass

    @abstractmethod
    def add_records(self, records):
        """
        Add record(s) to dataset and return modified dataset.

        """
        pass

    @abstractmethod
    def replace(self, record_in, record_out):
        """
        Replace a row with a given row.

        """
        pass

    @abstractmethod
    def create_subsets(self, n, sample_size, drop_records=None):
        """
        Create a number of training datasets (sub-samples from main dataset)
        of a given sample size and with the option to remove some records.

        """
        pass

    @abstractmethod
    def __add__(self, other):
        """
        Adding two Dataset objects together.

        """
        pass

    @abstractmethod
    def __iter__(self):
        """
        Returns an iterator over records in the dataset.
        """
        pass

    @property
    def label(self):
        return "Unnamed dataset"


class RecordSetDataset(Dataset):
    """
    Abstract class to represent datasets that can be represented as an unordered
    collection of "records", where each user is associated with exactly one
    record (and vice versa).

    This class provides utility to manipulate such datasets, but does not make
    assumptions on what records are. Children classes will define these (see,
    e.g., .tabular.TabularDataset).

    This class assumes that the data can be represented internally as either a
    Pandas DataFrame or a Pandas Series. Note that the latter can contain
    arbitrary objects, so this should not be a problem in practice.

    """

    def __init__(self, data, description, RecordClass):
        """
        Create a record-set dataset.

        Parameters
        ----------
        data: pd.Series *or* pd.DataFrame.
        description: DataDescription object.
        RecordClass: subclass of self.__class__ to use for individual records.

        """
        self.data = data
        self.description = description
        self.RecordClass = RecordClass

    # All the .read and .write methods are not implemented by the abstract
    # class. Implement these for children classes.

    def sample(self, n_samples=1, frac=None, random_state=None):
        """
        Sample a set of records from a Dataset object.

        Parameters
        ----------
        n_samples : int
            Number of records to sample. If frac is not None, this parameter is ignored.

        frac : float
            Fraction of records to sample.

        random_state : optional
            Passed to `pandas.DataFrame.sample()`

        Returns
        -------
        Dataset
            A Dataset object with a sample of the records of the original object.

        """
        if frac:
            n_samples = int(frac * len(self))

        return self.__class__(
            data=self.data.sample(n_samples, random_state=random_state),
            description=self.description,
        )

    def get_records(self, record_ids):
        """
        Get a record from the Dataset object

        Parameters
        ----------
        record_ids : list[int]
            List of indexes of records to retrieve.

        Returns
        -------
        Dataset
            A Dataset object with the record(s).

        """
        if len(record_ids) == 1:
            return self.RecordClass(
                self.data.iloc[record_ids], self.description, record_ids[0]
            )
        return self.__class__(self.data.iloc[record_ids], self.description)

    def drop_records(self, record_ids=[], n=1, in_place=False):
        """
        Drop records from the Dataset object, if record_ids is empty it will drop a random record.

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
        Dataset or None
            A new Dataset object without the record(s) or None if in_place=True.

        """
        if len(record_ids) == 0:
            # drop n random records if none provided
            record_ids = np.random.choice(self.data.index, size=n).tolist()

        else:
            # TODO: the indices expected by pandas are the ones used by .loc,
            # whereas in this file we use mostly .iloc. This needs to be
            # cleaned in some way. At the moment, we renumber record_ids to
            # be absolute indices (in 0, ..., len(dataset)-1).
            record_ids = [self.data.index[i] for i in record_ids]

        new_data = self.data.drop(record_ids)

        if in_place:
            self.data = new_data
            return

        return self.__class__(new_data, self.description)

    def add_records(self, records, in_place=False):
        """
        Add record(s) to dataset and return modified dataset.

        Parameters
        ----------
        records : Dataset
            A Dataset object with the record(s) to add.
        in_place : bool
            Bool indicating whether or not to change the dataset in-place or return
            a copy. If True, the dataset is changed in-place. The default is False.

        Returns
        -------
        Dataset or None
            A new Dataset object with the record(s) or None if inplace=True.

        """

        if in_place:
            assert (
                self.description == records.description
            ), "Both datasets must have the same data description."

            self.data = pd.concat([self.data, records.data])
            return

        # If not in_place this does the same as __add__.
        return self.__add__(records)

    def replace(self, records_in, records_out=[], in_place=False):
        """
        Replace a record with another one in the dataset, if records_out is empty it will remove a random record.

        Parameters
        ----------
        records_in : Dataset
            A Dataset object with the record(s) to add.
        records_out : list(int)
            List of indexes of records to drop.
        in_place : bool
            Bool indicating whether or not to change the dataset in-place or return
            a copy. If True, the dataset is changed in-place. The default is False.

        Returns
        -------
        Dataset or None
            A modified Dataset object with the replaced record(s) or None if in_place=True..

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

    def create_subsets(self, n, sample_size, drop_records=False):
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
        list(Dataset)
            A lists containing subsets of the data with and without the target record(s).

        """
        assert sample_size <= len(
            self
        ), f"Cannot create subsets larger than original dataset, sample_size max: {len(self)} got {sample_size}"

        # Create splits.
        splits = index_split(self.data.shape[0], sample_size, n)

        # Returns a list of Dataset subsampled from this dataset.
        subsamples = [self.get_records(train_index) for train_index in splits]

        # If required, remove the indices from the dataset.
        if drop_records:
            for train_index in splits:
                self.drop_records(train_index, in_place=True)

        return subsamples

    def empty(self):
        """
        Create an empty Dataset with the same description as the current one.
        Short-hand for Dataset.get_records([]).

        Returns
        -------
        Dataset
            Empty tabular dataset.

        """
        return self.get_records([])

    def copy(self):
        """
        Create a Dataset that is a deep copy of this one. In particular,
        the underlying data is copied and can thus be modified freely.

        Returns
        -------
        Dataset
            A copy of this Dataset.

        """
        return self.__class__(self.data.copy(), self.description)

    def __add__(self, other):
        """
        Adding two Dataset objects with the same data description together

        Parameters
        ----------
        other : (Dataset)
            A Dataset object.

        Returns
        -------
        Dataset
            A Dataset object with the addition of two initial objects.

        """

        assert (
            self.description == other.description
        ), "Both datasets must have the same data description"

        return self.__class__(pd.concat([self.data, other.data]), self.description)

    def __iter__(self):
        """
        Returns an iterator over records in this dataset,

        Returns
        -------
        iterator
            An iterator object that iterates over individual records, as Records.

        """
        # Depending on the internal representation, a record is converted differently.
        if isinstance(self.data, pd.DataFrame):
            _convert_data = lambda record: record.to_frame().T
            data_iterator = self.data.iterrows()
        else:
            _convert_data = lambda record: pd.Series([record])
            data_iterator = self.data.items()
        # iterrows() returns tuples (index, record), and map applies a 1-argument
        # function to the iterable it is given, hence why we have idx_and_rec
        # instead of the cleaner (idx, rec).
        convert_record = lambda idx_and_rec: self.RecordClass(
            data=_convert_data(idx_and_rec[1]),
            description=self.description,
            identifier=idx_and_rec[0],
        )
        return map(convert_record, data_iterator)

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
        to be in a Dataset are the rows, treated as 1-row Datasets.

        Parameters
        ----------
        item : Object
            Object to check membership of.

        Returns
        -------
        bool
            Whether or not item is considered to be contained in self.

        """
        if not isinstance(item, self.__class__):
            raise ValueError(
                f"Only {str(self.__class__)} can be checked for containment, not {type(item)}."
            )
        if len(item) != 1:
            raise ValueError(
                f"Only length-1 Datasets can be checked for containment, got length {len(item)})."
            )
        # Check if a record matches the item (flattening the result for pd.DataFrame).
        are_records_equal = self.data == item.data.iloc[0]
        if len(are_records_equal.shape) > 1:
            are_records_equal = are_records_equal.all(axis=1)
        return are_records_equal.any()

    @property
    def label(self):
        return self.description.label


class Record(Dataset):
    """Generic class holding the data of a user record."""

    def __init__(self, data, description, identifier):
        """
        Create a Record (children of Dataset).

        Parameters
        ----------
        data: Dataset.
            Carried over from Dataset.
        description: Description
            Carried over from Dataset.
        identified: int or str
            A unique identifier for this record in the original dataset.

        """
        super().__init__(data, description)
        self.id = identifier

    def copy(self):
        """
        Create a Record that is a deep copy of this one. In particular,
        the underlying data is copied and can thus be modified freely.

        Returns
        -------
        Record: A copy of this Record.

        """
        return self.__class__(self.data.copy(), self.description, self.id)

    def get_id(self, dataset):
        """

        Check if the record is found on a given Dataset and return the object id (index) on that
        dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset that contains this record.

        Returns
        -------
        int
            The id of the object based on the index in the original dataset.

        """

        merged = pd.merge(dataset.data, self.data, how="outer", indicator=True)

        if merged[merged["_merge"] == "both"].shape[0] != 1:
            raise AssertionError(
                "Error, more than one copy of this record is present on the dataset"
            )

        return merged[merged["_merge"] == "both"].index.values[0]

    def set_id(self, identifier):
        """
        Overwrite the id attribute on the Record.

        Parameters
        ----------
        identifier: int or str
            An id value to be assigned to the Record id attribute

        Returns
        -------
        None

        """
        self.id = identifier
        self.data.index = pd.Index([identifier])

        return

    def __add__(self, other):
        if isinstance(other, Record):
            # This is not allowed: records added together create a dataset, and
            # thus the result is not of type self.__class__. We explicitly ask
            # you to cast this record to a dataset of relevant type.
            raise Exception(
                "Adding two records is not allowed. Convert either to Dataset first."
            )
        # The other is a Dataset, and so supports addition with a recod.
        return other.__add__(self)

    @property
    def label(self):
        """
        The label for records is their identifier. We assume here that the label
        of the rest of the dataset is obvious from context. If not, it can be
        retrived as self.description.label.

        """
        return str(self.id)


class DataDescription:
    """
    Describes the content of a record (and a dataset).

    By default, this just holds the label for the dataset (and subsets of the
    datasets), and is shared by all datasets from the same source. Additional
    functionalities can be added in children classes.

    """

    def __init__(self, label=None):
        """
        Parameters
        ----------
        label: str (optional)
            The name to use to describe this dataset in reports.
        """
        self._label = label or "Unnamed dataset"

    def __eq__(self, other_description):
        """
        Check that the descriptions are equal. Since there is no data, this is
        based only on label.

        """
        if not isinstance(other_description, DataDescription):
            return False
        return self.label == other_description.label

    @property
    def label(self):
        """
        A label that describes the underlying dataset (and children).

        """
        return self._label
