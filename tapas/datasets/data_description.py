"""
A represention of the metadata describing a dataset.
"""

# TODO: this is only for tabular data?

class DataDescription:

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
        self.schema = schema
        self._label = label or "Unnamed dataset"

    def view(self, columns):
        """
        Returns the same DataDescription restricted to a subset of columns.

        """
        return DataDescription([c for c in self.schema if (c['name'] in columns)], self.label)

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

    def __eq__(self, other_dataset_description):
        """
        Check that the content of the description (schema) is identical. Label is isgnored.

        """
        if not isinstance(other_dataset_description, DataDescription):
            return False
        return self.schema == other_dataset_description.schema

    @property
    def label(self):
        """
        A label that describes the underlying dataset (and children).

        """
        return self._label
    
