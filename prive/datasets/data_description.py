

class DataDescription:
    def __init__(self, description):
        self.description = description

    @property
    def num_features(self):
        """
        int : Number of columns.

        """
        return len(self.description)

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

            # If it's representation is a string, it's not finite, so can't be one-hot encoded
            elif isinstance(cdict['representation'], str):
                nfeatures += 1

            # If it's representation is an int, then the int represents how many
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
        Can be indexed either by an int, indicating the index of the column, or
        by a string, indicating the name of the column.

        """
        if isinstance(key, int):
            return self.description[key]

        elif isinstance(key, str):
            for cdict in self.description:
                if cdict['name'] == key:
                    return cdict
            raise KeyError(f'No column named {key}')

        else:
            raise KeyError(f'Key must be an int or str, got {type(key)}')

    def __repr__(self):
        return 'Data Description\n' + f'Columns: {list(cdict["name"] for cdict in self)}'