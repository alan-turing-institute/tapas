from prive.datasets import TabularDataset
import pandas as pd

class TabularRecord(TabularDataset):
    """
     Class for tabular record object. The tabular data is a Pandas Dataframe with 1 row
     and the data description is a dictionary.

     """

    def __init__(self, data, description, id):
        super().__init__(data, description)
        # id of the object based on their index on the original dataset
        self.id = id

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
        if tabular_row.data.shape[0]!=1:
            raise AssertionError(f'Parent TabularDataset object must contain only 1 record, not {tabular_row.data.shape[0]}')

        return cls(tabular_row.data, tabular_row.description,tabular_row.data.index.values[0])

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

        merged = pd.merge(tabular_dataset.data, self.data, how='outer', indicator=True)

        if merged[merged['_merge']=='both'].shape[0] != 1:
            raise AssertionError('Error, more than one copy of this record is present on the dataset')

        return merged[merged['_merge'] == 'both'].index.values[0]

    def set_id(self, id):
        """

        Overwrite the id attribute on the TabularRecord object.

        Parameters
        ----------
        id: int or str
            An id value to be assigned to the TabularRecord id attribute

        Returns
        -------
        None

        """
        self.id = id
        self.data.index = pd.Index([id])

        return

