from .groundhog import Groundhog
from .set_classifiers import SetReprClassifier, LRClassifier, RFClassifier, NaiveRep


def load_attack(name, params, dataset):
    if name == 'Groundhog':
        # Determine which setrep is being used
        if params['setrep'] == 'NaiveRep':
            setrep = NaiveRep

        # Determine classifier
        if params['classifier'] == 'LogisticRegression':
            final_classifier = LRClassifier

        elif params['classifier'] == 'RandomForest':
            final_classifier = RFClassifier

        classifier = SetReprClassifier(setrep, final_classifier, dataset.description)
        attack = Groundhog(classifier, dataset.description)

    elif name == 'ClosestDistance':
        # Do something here for CD attack
        raise NotImplementedError

    return attack
