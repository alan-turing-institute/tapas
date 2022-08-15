from .groundhog import GroundhogAttack
from .closest_distance import ClosestDistanceMIA

from .set_classifiers import FeatureBasedSetClassifier, NaiveSetFeature, HistSetFeature

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# TODO: do we need this?


def load_attack(name, params, dataset):
    if name == "Groundhog":
        # Determine which set features to use.
        if params["setrep"] == "Naive":
            setrep = NaiveSetFeature()
        elif params["setrep"] == "Hist":
            setrep = HistSetFeature()
        elif params['setrep'] == "Groundhog":
            setrep = NaiveSetFeature() + HistSetFeature()

        # Determine classifier to use.
        if params["classifier"] == "LogisticRegression":
            final_classifier = LogisticRegression()

        elif params["classifier"] == "RandomForest":
            final_classifier = RandomForestClassifier()

        # Combine these
        classifier = FeatureBasedSetClassifier(
            setrep, final_classifier, dataset.description
        )
        attack = GroundhogAttack(classifier, dataset.description)

    elif name == "ClosestDistance":
        # Do something here for CD attack
        threshold = params.get("threshold", None)
        fpr = params.get("fpr", None)
        tpr = params.get("tpr", None)
        convert = lambda s: None if s in None else float(s)
        attack = ClosestDistanceMIA(
            fpr=convert(fpr), tpr=convert(tpr), threshold=convert(threshold),
        )

    return attack
