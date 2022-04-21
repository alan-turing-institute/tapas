from prive.datasets import TabularDataset


class TabularRecord(TabularDataset):
    """
     Class for tabular record object. The tabular data is a Pandas Dataframe with 1 row
     and the data description is a dictionary.

     """

    def __init__(self, data, description):
        super().__init__(data, description)

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

        return cls(tabular_row.data, tabular_row.description)
