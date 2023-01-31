"""
A network description class
"""


class TUDatasetDescription:

    def __init__(self, label=None):
        """
        Parameters
        ----------
        label: str (optional)
            The name to use to describe this dataset in reports.
        """
        self._label = label or "Unnamed dataset"

    def __eq__(self, other_network_description):
        """
        Check that the content of the description (schema) is identical. Label is ignored.

        """
        if not isinstance(other_network_description, TUDatasetDescription):
            return False
        return True

    @property
    def label(self):
        """
        A label that describes the underlying dataset (and children).

        """
        return self._label
