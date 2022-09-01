"""
Threat Models describe the assumptions under which an attack takes place.

Threat Models are composed of three elements:
 1. What the attacker aims to infer (e.g., membership, attribute).
 2. What the attacker knows about the generator (no-, black-, white-box).
 3. What the attacker knows about the training dataset.

A threat model thus describes what an attack wants to predict, and how it
can do that. It also describes how to _evaluate_ the success of an attack.
For instance, for a black-box membership inference attack with auxiliary data,
the attacker is able to run the generator on datasets samples from the
auxiliary data, which may or may not contain a target record. The evaluation
of the attack is performed on datasets from a disjoint dataset (test set),
from which training datasets are sampled, with or without the target record.

"""

from abc import ABC, abstractmethod

# For
import pickle


class ThreatModel(ABC):
    """
    Abstract base class for a threat model.

    A threat model describes the conditions under which an attack takes place:
     * What the attacker is trying to learn.
     * What the attacker knows about the method.
     * What the attacker knows about the training dataset.

    """

    @abstractmethod
    def test(self, attack, *args, **kwargs):
        """
        This method should implement the logic for testing an attack's success
        against the prescribed threat model. It takes as argument an Attack
        object, as well as (potential) additional parameters.

        """

        abstract

    @classmethod
    def load(cls, name):
        """
        Load a ThreatModel saved with self.save(name).

        Parameters
        ----------
        name: str
            The prefix of the filename (`name`.pkl) to which the threat model
            was saved.

        """
        filename = name + ".pkl"
        with open(filename, "rb") as ff:
            threat_model = pickle.load(ff)
        # Check that the object retrieved from disk is a threat model.
        if not isinstance(threat_model, ThreatModel):
            raise Exception("Pickled object is not a ThreatModel.")
        # Set the name of the threat model.
        threat_model._name = name
        return threat_model

    def save(self, name=None):
        """
        Save a copy of this ThreatModel, including all internal variables.

        Parameters
        ----------
        name: str (default None)
            The prefix of the filename (`name`.pkl) to which this threat model
            is saved. If self.name is None, then this attempts to use self._name,
            which is set exclusively by ThreatModel.load(name).

        """
        if name is None:
            if hasattr(self, "_name"):
                name = self._name
            else:
                raise Exception("Name is required to save this ThreatModel.")
        else:
            self._name = name
        filename = name + ".pkl"
        with open(filename, "wb") as ff:
            pickle.dump(self, ff)


class TrainableThreatModel(ThreatModel):
    """
    Some threat models additionally define a way to train attacks with
    synthetic datasets generated using the attacker's knowledge.

    """

    def generate_training_samples(self, num_samples):
        """
        Generate synthetic datasets (and potentially associated labels) to
        train an attack.

        """
