"""Classes to process tabular datasets."""

import io
import json
import pandas as pd

from .dataset import RecordSetDataset, Record, DataDescription
from .utils import encode_data, get_dtype


# Dataset description file: specifies the type of the different attributes.

class TabularDataDescription(DataDescription):

    def __init__(self, schema, label = None):
        """
        Parameters
        ----------
        schema: list[dict]
            A list of metadata about each column. Each column is represented by a dictionary whose
            values are the ``name``, ``type``, and on-disk ``representation`` of the column.
        label: str (optional)
            The name to use to describe this dataset in reports.
        """
        DataDescription.__init__(self, label)
        self.schema = schema

    def view(self, columns):
        """
        Returns the same DataDescription restricted to a subset of columns.

        """
        return TabularDataDescription([c for c in self.schema if (c['name'] in columns)], self.label)

    @property
    def num_features(self):
        """
        int : Number of columns.

        """
        return len(self.schema)

    @property
    def columns(self):
        """
        tuple: name of all columns, in order.

        """
        return tuple([column['name'] for column in self.schema])
    

    @property
    def encoded_dim(self):
        """
        int : Number of dimensions the data would have if encoded. This assumes
        ordered and infinite variables will have one dimension, and only finite,
        unordered variables would be one-hot encoded, where they will require
        one dimension per category.

        """
        nfeatures = 0

        for cdict in self:
            # If it's ordered it will be encoded as a single int
            if 'ordered' in cdict['type']:
                nfeatures += 1

            # If its representation is a string, it's not finite, so can't be one-hot encoded
            elif isinstance(cdict['representation'], str):
                nfeatures += 1

            # If its representation is an int, then the int represents how many
            # categories there are, and we already checked that it's not ordered
            # so it must be categorical, and therefore will be one-hot encoded
            elif isinstance(cdict['representation'], int):
                nfeatures += cdict['representation']

            # If it's not ordered and not one of the above, it must be a list
            # of possible values. The length indicates how many dimensions
            # one-hot encoding it will require.
            else:
                nfeatures += len(cdict['representation'])

        return nfeatures

    @property
    def one_hot_cols(self):
        """
        list[str] : List of column names that would be one-hot encoded if encoded.

        """
        cols = []
        for cdict in self:
            if cdict['type'] == 'finite':
                cols.append(cdict['name'])
        return cols

    def __getitem__(self, key):
        """
        Can be indexed either by an int, indicating the index of the column, or by a string,
        indicating the name of the column. If a string, returns the first item whose ``name`` matches
        ``key``.

        """
        if isinstance(key, int):
            return self.schema[key]

        elif isinstance(key, str):
            for cdict in self.schema:
                if cdict['name'] == key:
                    return cdict
            raise KeyError(f'No column named {key}')

        else:
            raise KeyError(f'Key must be an int or str, got {type(key)}')

    def __repr__(self):
        return 'Data Description\n' + f'Columns: {list(cdict["name"] for cdict in self)}'

    def __eq__(self, other_description):
        """
        Check that the content of the description (schema) is identical. Label is isgnored.

        """
        if not isinstance(other_description, TabularDataDescription):
            return False
        # Check equality of schemas rather than the label.
        return self.schema == other_description.schema
    


# Helper functions for parsing file-like objects.

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

    description = TabularDataDescription(schema, label=label)
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



class TabularDataset(RecordSetDataset):
    """
    Class to represent tabular data as a Dataset. Internally, the tabular data
    is stored as a Pandas Dataframe and the schema is an array of types.

    """

    def __init__(self, data, description):
        """
        Parameters
        ----------
        data: pandas.DataFrame (or a valid argument for pandas.DataFrame).
        description: tapas.tabular.TabularDataDescription
        label: str (optional)

        """
        # Convert the input to a Pandas DataFrame (internal representation).
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns = [c['name'] for c in description])
        # Check that the description is for a tabular dataset.
        assert isinstance(description, TabularDataDescription), 'description needs to be of class TabularDataDescription'
        # Init the parent class (mostly used to set the recorrd class).
        RecordSetDataset.__init__(self, data, description, TabularRecord)


    # Methods to read and write tabular datasets.

    @classmethod
    def read_from_string(cls, data, description):
        """
        Parameters
        ----------
        data: str
          The csv version of the data
        description: TabularDataDescription

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


    # Methods specific to tabular data.

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
    


class TabularRecord(Record, TabularDataset):
    """
    Class for tabular record object. The tabular data is a Pandas Dataframe with 1 row
    and the data description is a dictionary.

    """

    def __init__(self, data, description, identifier):
        Record.__init__(self, data, description, identifier)
        TabularDataset.__init__(self, data, description)

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
