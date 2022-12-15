"""Classes to represent the data object"""
from abc import ABC, abstractmethod
import json
import io

import numpy as np
import pandas as pd

from .data_description import DataDescription
from .utils import encode_data, index_split, get_dtype


# Helper function for parsing file-like objects
def _parse_csv(fp, schema, label=None):
    """
    Parse fp into a TabularDataset using schema

    Parameters
    ----------
    fp: A file-type object
    schema: A json schema
    label: a name to represent this dataset (optional)

    Returns
    -------
    TabularDataset

    """
    ## read_csv does not accept datetime in the dtype argument, so we read dates as strings and
    ## then convert them.
    dtypes = {
        i: get_dtype(col["type"], col["representation"]) for i, col in enumerate(schema)
    }

    cnames = [col["name"] for col in schema]
    
    data = pd.read_csv(fp, header=validate_header(fp, cnames), dtype=dtypes, index_col=None, names=cnames)    
    
    ## Convert any date or datetime fields to datetime.
    for c in [
        col["name"]
        for col in schema
        if col["representation"] == "date" or col["representation"] == "datetime"
    ]:
        data[c] = pd.to_datetime(data[c])

    description = DataDescription(schema, label=label)
    return TabularDataset(data, description)


def validate_header(fp, cnames):
    """
    Helper function to toggle 'header' argument in pd.read_csv()
    
    Reads first row of data. 
    
    Raises exception is header exists and it does not match schema.
    
    

    Parameters
    ----------
    fp: A file-type object
    cnames: Column names from schema.

    Returns
    -------
    an option for 'header' argument in pd.read_csv(). 
    
    0 if header exists and it matches cnames.
    None is header does not exist. 

    """
    if isinstance(fp, io.StringIO):
        fp = io.StringIO(fp.getvalue())
    
    row0 = pd.read_csv(fp, header=None, index_col=None, nrows=1) 
    if all(row0.iloc[0].apply(lambda x: isinstance(x, str))):
        # is a potential header row
        if (row0.iloc[0] == cnames).all(): 
            # is the same.
            return 0
        else:
            # is a header row but invalid.
            invalid=[{'data': r, 'schema': c} for r, c in zip(row0.iloc[0], cnames) if r != c]
            raise AssertionError(
                f"Data has header row that does not match schema. Invalid matches:\n {invalid}"
            )
    else:
        # is not a header row. 
        return None
        

## Classes
## -------


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
    


class TabularDataset(Dataset):
    """
    Class to represent tabular data as a Dataset. Internally, the tabular data
    is stored as a Pandas Dataframe and the schema is an array of types.

    """

    def __init__(self, data, description):
        """
        Parameters
        ----------
        data: pandas.DataFrame (or a valid argument for pd.DataFrame).
        description: tapas.datasets.data_description.DataDescription
        label: str (optional)
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns = [c['name'] for c in description])
        self.data = data
        
        assert isinstance(description,DataDescription), 'description needs to be of class DataDescription'
        self.description = description

    @classmethod
    def read_from_string(cls, data, description):
        """
        Parameters
        ----------
        data: str
          The csv version of the data
        description: DataDescription

        Returns
        -------
        TabularDataset
        """
        return _parse_csv(io.StringIO(data), description.schema, description.label)

    @classmethod
    def read(cls, filepath, label = None):
        """
        Read csv and json files for dataframe and schema respectively.

        Parameters
        ----------
        filepath: str
            Full path to the csv and json, excluding the ``.csv`` or ``.json`` extension.
            Both files should have the same root name.
        label: str or None
            An optional string to represent this dataset.

        Returns
        -------
        TabularDataset
            A TabularDataset.

        """
        with open(f"{filepath}.json") as f:
            schema = json.load(f)

        return _parse_csv(f"{filepath}.csv", schema, label or filepath)

    def write_to_string(self):
        """
        Return a string holding the dataset (as a csv).

        """
        # Passing None to to_csv returns the csv as a string
        return self.data.to_csv(None, index=False)

    def write(self, filepath):
        """
        Write data and description to file

        Parameters
        ----------
        filepath : str
            Path where the csv and json file are saved.

        """

        with open(f"{filepath}.json", "w") as fp:
            json.dump(self.description.schema, fp, indent=4)

        # TODO: Make sure this writes it exactly as needed
        self.data.to_csv(filepath + ".csv", index=False)

    def sample(self, n_samples=1, frac=None, random_state=None):
        """
        Sample a set of records from a TabularDataset object.

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
        TabularDataset
            A TabularDataset object with a sample of the records of the original object.

        """
        if frac:
            n_samples = int(frac * len(self))

        return TabularDataset(
            data=self.data.sample(n_samples, random_state=random_state),
            description=self.description,
        )

    def get_records(self, record_ids):
        """
        Get a record from the TabularDataset object

        Parameters
        ----------
        record_ids : list[int]
            List of indexes of records to retrieve.

        Returns
        -------
        TabularDataset
            A TabularDataset object with the record(s).

        """

        # TODO: what if the index is supposed to be a column? an identifier?
        if len(record_ids) == 1:
            return TabularRecord(
                self.data.iloc[record_ids], self.description, record_ids[0]
            )
        return TabularDataset(self.data.iloc[record_ids], self.description)

    def drop_records(self, record_ids=[], n=1, in_place=False):
        """
        Drop records from the TabularDataset object, if record_ids is empty it will drop a random record.

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
        TabularDataset or None
            A new TabularDataset object without the record(s) or None if in_place=True.

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

        return TabularDataset(new_data, self.description)

    def add_records(self, records, in_place=False):
        """
        Add record(s) to dataset and return modified dataset.

        Parameters
        ----------
        records : TabularDataset
            A TabularDataset object with the record(s) to add.
        in_place : bool
            Bool indicating whether or not to change the dataset in-place or return
            a copy. If True, the dataset is changed in-place. The default is False.

        Returns
        -------
        TabularDataset or None
            A new TabularDataset object with the record(s) or None if inplace=True.

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
        records_in : TabularDataset
            A TabularDataset object with the record(s) to add.
        records_out : list(int)
            List of indexes of records to drop.
        in_place : bool
            Bool indicating whether or not to change the dataset in-place or return
            a copy. If True, the dataset is changed in-place. The default is False.

        Returns
        -------
        TabularDataset or None
            A modified TabularDataset object with the replaced record(s) or None if in_place=True..

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
        Create an empty TabularDataset with the same description as the current one.
        Short-hand for TabularDataset.get_records([]).

        Returns
        -------
        TabularDataset
            Empty tabular dataset.

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
        return TabularDataset(self.data.copy(), self.description)

    def view(self, columns = None, exclude_columns = None):
        """
        Create a TabularDataset object that contains a subset of the columns of
        this TabularDataset. The resulting object only has a copy of the data,
        and can thus be modified without affecting the original data.

        Parameters
        ----------
        Exactly one of `columns` and `exclude_columns` must be defined.

        columns: list, or None
            The columns to include in the view.
        exclude_columns: list, or None
            The columns to exclude from the view, with all other columns included.

        Returns
        -------
        TabularDataset
            A subset of this data, restricted to some columns.

        """
        assert (
            columns is not None or exclude_columns is not None
        ), "Empty view: specify either columns or exclude_columns."
        assert (
            columns is None or exclude_columns is None
        ), "Overspecified view: only one of columns and exclude_columns can be given."

        if exclude_columns is not None:
            columns = [c for c in self.description.columns if c not in exclude_columns]

        return TabularDataset(self.data[columns], self.description.view(columns))

    @property
    def as_numeric(self):
        """
        Encodes this dataset as a np.array, where numeric values are kept as is
        and categorical values are 1-hot encoded. This is only computed once
        (for efficiency reasons), so beware of modifying TabularDataset after
        using this property.

        The columns are kept in the order of the description, with categorical
        variables encoded over several contiguous columns.

        Returns
        -------
        np.array

        """
        return encode_data(self)

    def __add__(self, other):
        """
        Adding two TabularDataset objects with the same data description together

        Parameters
        ----------
        other : (TabularDataset)
            A TabularDataset object.

        Returns
        -------
        TabularDataset
            A TabularDataset object with the addition of two initial objects.

        """

        assert (
            self.description == other.description
        ), "Both datasets must have the same data description"

        return TabularDataset(pd.concat([self.data, other.data]), self.description)

    def __iter__(self):
        """
        Returns an iterator over records in this dataset,

        Returns
        -------
        iterator
            An iterator object that iterates over individual records, as TabularRecords.

        """
        # iterrows() returns tuples (index, record), and map applies a 1-argument
        # function to the iterable it is given, hence why we have idx_and_rec
        # instead of the cleaner (idx, rec).
        convert_record = lambda idx_and_rec: TabularRecord.from_dataset(
            TabularDataset(
                # iterrows() outputs pd.Series rather than .DataFrame, so we convert here:
                data=idx_and_rec[1].to_frame().T,
                description=self.description,
            )
        )
        return map(convert_record, self.data.iterrows())

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
        to be in a TabularDataset are the rows, treated as 1-row TabularDatasets.

        Parameters
        ----------
        item : Object
            Object to check membership of.

        Returns
        -------
        bool
            Whether or not item is considered to be contained in self.

        """
        if not isinstance(item, TabularDataset):
            raise ValueError(
                f"Only TabularDatasets can be checked for containment, not {type(item)}"
            )
        if len(item) != 1:
            raise ValueError(
                f"Only length-1 TabularDatasets can be checked for containment, got length {len(item)})"
            )

        return (self.data == item.data.iloc[0]).all(axis=1).any()

    @property
    def label(self):
        return self.description.label
    


class TabularRecord(TabularDataset):
    """
    Class for tabular record object. The tabular data is a Pandas Dataframe with 1 row
    and the data description is a dictionary.

    """

    def __init__(self, data, description, identifier):
        super().__init__(data, description)
        # id of the object based on their index on the original dataset
        self.id = identifier

    @classmethod
    def from_dataset(cls, tabular_row):
        """
        Create a TabularRecord object from a TabularDataset object containing 1 record.

        Parameters
        ----------
        tabular_row: TabularDataset
            A TabularDataset object containing one record.

        Returns
        -------
        TabularRecord
            A TabularRecord object

        """
        if tabular_row.data.shape[0] != 1:
            raise AssertionError(
                f"Parent TabularDataset object must contain only 1 record, not {tabular_row.data.shape[0]}"
            )

        return cls(
            tabular_row.data, tabular_row.description, tabular_row.data.index.values[0]
        )

    def get_id(self, tabular_dataset):
        """

        Check if the record is found on a given TabularDataset and return the object id (index) on that
        dataset.

        Parameters
        ----------
        tabular_dataset: TabularDataset
            A TabularDataset object.

        Returns
        -------
        int
            The id of the object based on the index in the original dataset.

        """

        merged = pd.merge(tabular_dataset.data, self.data, how="outer", indicator=True)

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
        return TabularRecord(self.data.copy(), self.description, self.id)

    @property
    def label(self):
        """
        The label for records is their identifier. We assume here that the label
        of the rest of the dataset is obvious from context. If not, it can be
        retrived as self.description.label.

        """
        return str(self.id)
